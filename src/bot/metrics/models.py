"""Типизированные записи метрик, попадающие в append-only JSONL и structlog.

Записи несут только счётчики и идентификаторы: содержимое сообщений,
команд и ответов модели в метрики не попадает никогда.
"""

from dataclasses import dataclass
from typing import Protocol

from bot.domain.ids import ModelId, RequestId

# Ключи-дубликаты structlog-события: kind и timestamp в лог не пишутся
# (structlog ставит свой timestamp, а kind — имя события).
_NON_LOG_FIELDS = frozenset({"kind", "timestamp"})


class MetricRecord(Protocol):
    """Запись метрики, пригодная и для JSONL, и для structlog-дубликата."""

    @property
    def kind(self) -> str: ...

    @property
    def log_event(self) -> str: ...

    def to_payload(self) -> dict[str, str | int | float | bool | None]: ...


def strip_log_fields(payload: dict[str, str | int | float | bool | None]) -> dict[str, object]:
    """Поля записи для structlog-дубликата (без kind и timestamp)."""
    return {key: value for key, value in payload.items() if key not in _NON_LOG_FIELDS}


@dataclass(frozen=True)
class LlmCallRecord:
    """Один вызов модели: шаг запуска, токены, латентность, оценка стоимости."""

    timestamp: str
    request_id: RequestId
    model: ModelId
    step: int
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: int
    estimated_cost: float | None

    @property
    def kind(self) -> str:
        return "llm_call"

    @property
    def log_event(self) -> str:
        return "metrics_llm_call"

    def to_payload(self) -> dict[str, str | int | float | bool | None]:
        return {
            "kind": self.kind,
            "timestamp": self.timestamp,
            "request_id": str(self.request_id),
            "model": str(self.model),
            "step": self.step,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_ms": self.latency_ms,
            "estimated_cost": self.estimated_cost,
        }


@dataclass(frozen=True)
class RunRecord:
    """Итог одного Запуска агента: шаги, суммарные токены, длительность, успех."""

    timestamp: str
    request_id: RequestId
    model: ModelId | None
    steps: int
    prompt_tokens: int | None
    completion_tokens: int | None
    duration_ms: int | None
    success: bool
    estimated_cost: float | None

    @property
    def kind(self) -> str:
        return "run"

    @property
    def log_event(self) -> str:
        return "metrics_run_finished"

    def to_payload(self) -> dict[str, str | int | float | bool | None]:
        return {
            "kind": self.kind,
            "timestamp": self.timestamp,
            "request_id": str(self.request_id),
            "model": self.model,
            "steps": self.steps,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "estimated_cost": self.estimated_cost,
        }
