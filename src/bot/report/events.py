"""Чтение JSONL-метрик в типизированные события.

Одна строка файла — одна запись, написанная ``MetricsRecorder``. Битые строки
(некорректный JSON, неизвестный kind, поля не тех типов) пропускаются: отчёт —
наблюдаемый артефакт, а не процесс, и не должен падать на испорченной строке.
"""

import json
from pathlib import Path

from bot.domain.ids import ModelId, RequestId
from bot.metrics.models import LlmCallRecord, RunRecord, ToolCallRecord

type MetricEvent = LlmCallRecord | ToolCallRecord | RunRecord


def read_events(path: Path) -> tuple[MetricEvent, ...]:
    """Прочитать события из JSONL; отсутствующий файл означает пустой отчёт."""
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ()
    events: list[MetricEvent] = []
    for line in raw_lines:
        event = _parse_line(line)
        if event is not None:
            events.append(event)
    return tuple(events)


def _parse_line(line: str) -> MetricEvent | None:
    try:
        payload = json.loads(line)
        kind = payload["kind"]
        # Записи без идентификатора запуска и временной метки не интерпретируемы.
        if not _str(payload, "request_id") or not _str(payload, "timestamp"):
            return None
        if kind == "llm_call":
            return _llm_call(payload)
        if kind == "tool_call":
            return _tool_call(payload)
        if kind == "run":
            return _run(payload)
        return None
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _llm_call(payload: dict[object, object]) -> LlmCallRecord:
    return LlmCallRecord(
        timestamp=_str(payload, "timestamp"),
        request_id=RequestId(_str(payload, "request_id")),
        model=ModelId(_str(payload, "model")),
        step=_int(payload, "step") or 0,
        prompt_tokens=_opt_int(payload, "prompt_tokens"),
        completion_tokens=_opt_int(payload, "completion_tokens"),
        repeated_context_tokens=_opt_int(payload, "repeated_context_tokens"),
        latency_ms=_int(payload, "latency_ms") or 0,
        estimated_cost=_opt_float(payload, "estimated_cost"),
    )


def _tool_call(payload: dict[object, object]) -> ToolCallRecord:
    return ToolCallRecord(
        timestamp=_str(payload, "timestamp"),
        request_id=RequestId(_str(payload, "request_id")),
        tool_name=_str(payload, "tool_name"),
        input_size=_int(payload, "input_size") or 0,
        output_size=_int(payload, "output_size") or 0,
        output_tokens=_int(payload, "output_tokens") or 0,
        duration_ms=_int(payload, "duration_ms") or 0,
        succeeded=_bool(payload, "succeeded"),
    )


def _run(payload: dict[object, object]) -> RunRecord:
    raw_model = _opt_str(payload, "model")
    return RunRecord(
        timestamp=_str(payload, "timestamp"),
        request_id=RequestId(_str(payload, "request_id")),
        model=ModelId(raw_model) if raw_model is not None else None,
        steps=_int(payload, "steps") or 0,
        prompt_tokens=_opt_int(payload, "prompt_tokens"),
        completion_tokens=_opt_int(payload, "completion_tokens"),
        repeated_context_tokens=_opt_int(payload, "repeated_context_tokens"),
        repeated_context_ratio=_opt_float(payload, "repeated_context_ratio"),
        duration_ms=_opt_int(payload, "duration_ms"),
        success=_bool(payload, "success"),
        estimated_cost=_opt_float(payload, "estimated_cost"),
    )


def _str(payload: dict[object, object], key: str) -> str:
    value = payload.get(key)
    return value if isinstance(value, str) else ""


def _opt_str(payload: dict[object, object], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) else None


def _int(payload: dict[object, object], key: str) -> int | None:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _opt_int(payload: dict[object, object], key: str) -> int | None:
    return _int(payload, key)


def _opt_float(payload: dict[object, object], key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _bool(payload: dict[object, object], key: str) -> bool:
    value = payload.get(key)
    return value if isinstance(value, bool) else False
