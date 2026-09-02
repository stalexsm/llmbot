"""CLI headless-прогона бенчмарка: ``uv run python -m bot.benchmark``.

Печатает по каждой задаче success/токены и сводку по набору. Прогон идёт
напрямую через агентный цикл против живого Ollama, без Telegram; метрики
запусков пишутся в общий JSONL и попадают в отчёт ``bot.report``.
"""

import argparse
import asyncio
import contextlib
import logging
import sys

import httpx
import structlog

from bot.agent.exec import ExecTool
from bot.agent.loop import AgentLoop
from bot.agent.prompts import build_system_prompt
from bot.agent.skills import load_skills, render_skills_index
from bot.benchmark.runner import BenchmarkRunner, BenchmarkSummary
from bot.benchmark.settings import BenchmarkSettings
from bot.benchmark.tasks import (
    EXPECTED_KIND_COUNTS,
    REPO_ROOT,
    benchmark_tasks,
    task_kind_counts,
)
from bot.benchmark.tracker import TokenTrackingProvider
from bot.config.settings import Settings
from bot.domain.ids import ModelId
from bot.inference.ollama import OllamaInferenceProvider
from bot.metrics.collector import RunMetricsCollector
from bot.metrics.provider import MeteredInferenceProvider
from bot.metrics.recorder import METRICS_DIRECTORY, MetricsRecorder
from bot.metrics.tool import MeteredTool


def configure_logging() -> structlog.stdlib.BoundLogger:
    """Тихий консольный лог: детали прогоне пишут structlog-события в stderr."""
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    return structlog.stdlib.get_logger()


async def run_benchmark(settings: Settings, logger: structlog.stdlib.BoundLogger) -> int:
    """Собрать граф агента без Telegram и прогнать набор задач. Код возврата — 0/1."""
    tasks = benchmark_tasks()
    counts = task_kind_counts(tasks)
    if counts != EXPECTED_KIND_COUNTS:
        logger.error("benchmark_task_set_mismatch", counts={str(k): v for k, v in counts.items()})
        return 1

    http_client = httpx.AsyncClient(timeout=settings.ollama_timeout_seconds, trust_env=False)
    try:
        ollama = OllamaInferenceProvider(
            client=http_client,
            base_url=settings.ollama_base_url,
            timeout_seconds=settings.ollama_timeout_seconds,
            logger=logger,
            think=settings.ollama_think,
        )
        tracker = TokenTrackingProvider(inner=ollama)
        collector = RunMetricsCollector(
            # Якорим к корню репозитория: прогон из другого cwd не должен
            # раскидывать метрики и рабочий каталог агента по разным местам.
            recorder=MetricsRecorder(directory=REPO_ROOT / METRICS_DIRECTORY, logger=logger),
            input_price_per_mtok=settings.metrics_input_price_per_mtok,
            output_price_per_mtok=settings.metrics_output_price_per_mtok,
        )
        inference = MeteredInferenceProvider(inner=tracker, collector=collector)
        skills = load_skills(settings.agent_skills_directory, logger)
        exec_tool = MeteredTool(
            inner=ExecTool(
                cwd=REPO_ROOT,
                timeout_seconds=settings.agent_exec_timeout_seconds,
                max_output_chars=settings.agent_exec_max_output_chars,
                logger=logger,
            ),
            collector=collector,
        )
        agent = AgentLoop(
            inference=inference,
            model=ModelId(settings.ollama_model),
            system_prompt=build_system_prompt(render_skills_index(skills)),
            tools=(exec_tool,),
            step_limit=settings.agent_max_steps,
            logger=logger,
        )
        runner = BenchmarkRunner(agent=agent, tracker=tracker, metrics=collector, logger=logger)
        summary = await runner.run_all(tasks)
    finally:
        await http_client.aclose()

    _print_results(summary)
    return 0


def _format_tokens(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}"


def _print_results(summary: BenchmarkSummary) -> None:
    lines: list[str] = ["=== Бенчмарк: по задачам ==="]
    for result in summary.results:
        kind = str(result.kind)
        success = "да" if result.success else "НЕТ"
        answer = "ок" if result.answer_correct else "неверно"
        line = (
            f"[{result.task_id}] {kind:<9} "
            f"success={success:<3} "
            f"answer={answer} "
            f"steps={result.steps_used} "
            f"tokens(in/out)={_format_tokens(result.prompt_tokens)}/"
            f"{_format_tokens(result.completion_tokens)} "
            f"time={result.duration_ms}ms"
        )
        if result.error:
            line += f" error={result.error}"
        lines.append(line)
    lines.append("=== Сводка ===")
    lines.append(
        f"Задач: {summary.total}, успехов: {summary.succeeded} "
        f"(success rate {summary.success_rate:.0%}), "
        f"верных ответов: {summary.answered}"
    )
    lines.append(
        "Средние токены на запуск: "
        f"input={_format_tokens(summary.avg_prompt_tokens)}, "
        f"output={_format_tokens(summary.avg_completion_tokens)}, "
        f"total={_format_tokens(summary.avg_total_tokens)}"
    )
    sys.stdout.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless-прогон бенчмарка против живого Ollama")
    parser.parse_args()
    logger = configure_logging()
    settings = BenchmarkSettings()  # pydantic-settings validates configuration
    exit_code = asyncio.run(run_benchmark(settings, logger))
    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
