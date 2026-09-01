"""Аккумулятор метрик по запускам: llm_call от декоратора, run — по завершении.

Состояние ключуется ``RequestId`` (run id = RequestId): декоратор на шве
инференса сообщает каждый вызов модели, а слой приложения закрывает запуск
``finish_run`` — тогда пишется агрегированная запись ``run``, суммы которой
сходятся с ``llm_call`` того же запуска.
"""

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from bot.domain.ids import ModelId, RequestId
from bot.inference.models import InferenceUsage
from bot.metrics.models import LlmCallRecord, RunRecord, ToolCallRecord, estimate_output_tokens
from bot.metrics.recorder import MetricsRecorder


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class _RunAccumulator:
    """Внутренний изменяемый аккумулятор одного запуска (не контракт)."""

    started_at: float = field(default_factory=time.monotonic)
    records: list[LlmCallRecord] = field(default_factory=list)


class RunMetrics(Protocol):
    """Порт закрытия запуска: потребляется приложением, реализуется метриками."""

    def finish_run(self, request_id: RequestId, *, success: bool) -> None: ...


class RunMetricsCollector:
    """Собирает вызовы модели по запускам и закрывает запуск записью run."""

    def __init__(
        self,
        recorder: MetricsRecorder,
        input_price_per_mtok: float,
        output_price_per_mtok: float,
    ) -> None:
        self._recorder = recorder
        self._input_price_per_mtok = input_price_per_mtok
        self._output_price_per_mtok = output_price_per_mtok
        self._runs: dict[RequestId, _RunAccumulator] = {}

    def record_llm_call(
        self,
        *,
        request_id: RequestId,
        model: ModelId,
        latency_ms: int,
        usage: InferenceUsage | None,
    ) -> None:
        """Записать один вызов модели и накинуть его в копилку запуска."""
        prompt_tokens = usage.prompt_tokens if usage is not None else None
        completion_tokens = usage.completion_tokens if usage is not None else None
        run = self._runs.setdefault(request_id, _RunAccumulator())
        record = LlmCallRecord(
            timestamp=_utc_now_iso(),
            request_id=request_id,
            model=model,
            step=len(run.records) + 1,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            estimated_cost=self._estimate_cost(prompt_tokens, completion_tokens),
        )
        run.records.append(record)
        self._recorder.record(record)

    def record_tool_call(
        self,
        *,
        request_id: RequestId,
        tool_name: str,
        input_size: int,
        output_size: int,
        duration_ms: int,
        succeeded: bool,
    ) -> None:
        """Записать один вызов инструмента с оценкой выходных токенов."""
        self._recorder.record(
            ToolCallRecord(
                timestamp=_utc_now_iso(),
                request_id=request_id,
                tool_name=tool_name,
                input_size=input_size,
                output_size=output_size,
                output_tokens=estimate_output_tokens(output_size),
                duration_ms=duration_ms,
                succeeded=succeeded,
            )
        )

    def finish_run(self, request_id: RequestId, *, success: bool) -> None:
        """Закрыть запуск: агрегированная запись run поверх его llm_call."""
        run = self._runs.pop(request_id, None)
        records = run.records if run is not None else []
        self._recorder.record(
            RunRecord(
                timestamp=_utc_now_iso(),
                request_id=request_id,
                model=records[-1].model if records else None,
                steps=len(records),
                prompt_tokens=self._sum_int(record.prompt_tokens for record in records),
                completion_tokens=self._sum_int(record.completion_tokens for record in records),
                duration_ms=(
                    int((time.monotonic() - run.started_at) * 1000) if run is not None else None
                ),
                success=success,
                estimated_cost=self._sum_float(record.estimated_cost for record in records),
            )
        )

    def _estimate_cost(
        self, prompt_tokens: int | None, completion_tokens: int | None
    ) -> float | None:
        """Оценка стоимости вызова; без токенов от провайдера стоимость неизвестна."""
        if prompt_tokens is None or completion_tokens is None:
            return None
        cost = (
            prompt_tokens / 1_000_000 * self._input_price_per_mtok
            + completion_tokens / 1_000_000 * self._output_price_per_mtok
        )
        return round(cost, 6)

    @staticmethod
    def _sum_int(values: Iterable[int | None]) -> int | None:
        """Сумма счётчиков токенов; None, если провайдер не сообщил ни одного."""
        known = [value for value in values if value is not None]
        if not known:
            return None
        return sum(known)

    @staticmethod
    def _sum_float(values: Iterable[float | None]) -> float | None:
        """Сумма стоимостей вызовов; None, если ни одна не была вычислена."""
        known = [value for value in values if value is not None]
        if not known:
            return None
        return sum(known)
