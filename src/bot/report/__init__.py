"""Отчёт по расходу токенов: текстовый dashboard и timeline запуска агента.

Источник — append-only JSONL метрик (``.data/metrics/events.jsonl``). Рендер —
чистая функция от событий: чтение файла (``events.py``) отделено от агрегации
(``aggregate.py``), построения timeline (``timeline.py``) и отрисовки текста
(``render.py``). Точка входа — ``uv run python -m bot.report``.
"""
