"""Headless-раннер бенчмарка: агентный цикл без Telegram.

Прогоняет задачи напрямую через ``AgentLoop`` против живого Ollama,
по каждой задаче собирает success и токены, по набору — сводку.
Success = финальный ответ в пределах лимита шагов без ошибки;
отдельно отмечается, прошёл ли ответ содержательную проверку задачи.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass

import structlog

from bot.agent.loop import AgentLoop
from bot.benchmark.tasks import BenchmarkTask, TaskKind
from bot.benchmark.tracker import TokenTrackingProvider
from bot.domain.ids import RequestId
from bot.domain.messages import InferenceMessage, MessageRole
from bot.metrics.collector import RunMetrics


@dataclass(frozen=True)
class TaskResult:
    """Итог одной задачи прогона: success, проверка ответа, токены."""

    task_id: str
    kind: TaskKind
    success: bool
    answer_correct: bool
    steps_used: int
    prompt_tokens: int | None
    completion_tokens: int | None
    duration_ms: int
    error: str | None = None


@dataclass(frozen=True)
class BenchmarkSummary:
    """Сводка по набору: success rate и средние токены на запуск."""

    results: tuple[TaskResult, ...]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def succeeded(self) -> int:
        return sum(1 for result in self.results if result.success)

    @property
    def success_rate(self) -> float:
        return self.succeeded / self.total if self.total else 0.0

    @property
    def answered(self) -> int:
        return sum(1 for result in self.results if result.answer_correct)

    def _average(self, values: Sequence[int | None]) -> float | None:
        known = [value for value in values if value is not None]
        if not known:
            return None
        return sum(known) / len(known)

    @property
    def avg_prompt_tokens(self) -> float | None:
        return self._average([result.prompt_tokens for result in self.results])

    @property
    def avg_completion_tokens(self) -> float | None:
        return self._average([result.completion_tokens for result in self.results])

    @property
    def avg_total_tokens(self) -> float | None:
        totals = [
            result.prompt_tokens + result.completion_tokens
            for result in self.results
            if result.prompt_tokens is not None and result.completion_tokens is not None
        ]
        return sum(totals) / len(totals) if totals else None


class BenchmarkRunner:
    """Прогоняет набор задач через агентный цикл и собирает результаты.

    Каждый запуск закрывается через порт ``RunMetrics``: записи ``run``
    bench-запусков попадают в общий JSONL, и dashboard считает их наравне
    с обычными запусками.
    """

    def __init__(
        self,
        agent: AgentLoop,
        tracker: TokenTrackingProvider,
        metrics: RunMetrics,
        logger: structlog.stdlib.BoundLogger,
    ) -> None:
        self._agent = agent
        self._tracker = tracker
        self._metrics = metrics
        self._logger = logger.bind(component="benchmark_runner")

    async def run_task(self, task: BenchmarkTask) -> TaskResult:
        request_id = RequestId(f"bench-{task.task_id}")
        started_at = time.monotonic()
        error: str | None = None
        steps_used = 0
        success = False
        answer = ""
        try:
            run = await self._agent.run(
                request_id,
                history=(),
                user_message=InferenceMessage(role=MessageRole.USER, content=task.prompt),
            )
            steps_used = run.steps_used
            if run.final_answer is not None and not run.stopped_by_limit:
                success = True
                answer = run.final_answer
        except Exception as exc:  # ошибка одной задачи не рвёт прогон набора
            error = f"{type(exc).__name__}: {exc}"
            self._logger.warning("benchmark_task_error", task_id=task.task_id, error=error)
        prompt_tokens, completion_tokens = self._tracker.totals(request_id)
        result = TaskResult(
            task_id=task.task_id,
            kind=task.kind,
            success=success,
            answer_correct=success and task.checker(answer),
            steps_used=steps_used,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration_ms=int((time.monotonic() - started_at) * 1000),
            error=error,
        )
        # Закрыть запуск агрегированной записью run — как в обычном чат-потоке.
        self._metrics.finish_run(request_id, success=result.success)
        return result

    async def run_all(self, tasks: Sequence[BenchmarkTask]) -> BenchmarkSummary:
        """Прогнать задачи последовательно: параллель исказила бы измерения."""
        results = tuple([await self._run_one(task) for task in tasks])
        return BenchmarkSummary(results=results)

    async def _run_one(self, task: BenchmarkTask) -> TaskResult:
        result = await self.run_task(task)
        self._logger.info(
            "benchmark_task_finished",
            task_id=result.task_id,
            kind=str(result.kind),
            success=result.success,
            answer_correct=result.answer_correct,
            steps=result.steps_used,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            duration_ms=result.duration_ms,
        )
        return result
