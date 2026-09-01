"""Append-only JSONL-хранилище метрик с structlog-дубликатами.

Одна строка ``events.jsonl`` — одна запись (``llm_call`` или ``run``).
Файл только дописывается: записи переживают рестарт и не перезаписываются.
Сбой диска метрики не рвёт — метрики наблюдаются, а не владеют процессом,
поэтому хранилище не бросает исключений наружу.
"""

import json
from pathlib import Path

import structlog

from bot.metrics.models import MetricRecord, strip_log_fields

EVENTS_FILENAME = "events.jsonl"


class MetricsRecorder:
    """Дописывает записи метрик в JSONL и дублирует их в structlog."""

    def __init__(self, directory: Path, logger: structlog.stdlib.BoundLogger) -> None:
        self._directory = directory
        self._logger = logger.bind(component="metrics_recorder")

    def record(self, record: MetricRecord) -> None:
        """Дописать запись одной строкой; сбой хранилища не рвёт работу бота."""
        payload = record.to_payload()
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            with (self._directory / EVENTS_FILENAME).open("a", encoding="utf-8") as events_file:
                events_file.write(line)
        except OSError:
            self._logger.warning("metrics_storage_failed", kind=record.kind)
        # structlog-дубль пишется независимо от судьбы файла: по request_id
        # он всегда коррелирует с JSONL-записью того же вызова.
        self._logger.info(record.log_event, **strip_log_fields(payload))
