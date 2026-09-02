# 02: Мониторинг LLM-вызовов и запись запуска (llm_call + run)

**What to build:** обычный диалог с ботом оставляет измеримый след: декоратор на шве `InferenceProvider` пишет событие `llm_call` на каждый вызов модели (run id = `RequestId`, модель, номер шага, input/output токены, латентность, estimated_cost), а по завершении Запуска агента пишется агрегированная запись `run` (шаги, суммарные токены, длительность, success). События — append-only JSONL в `.data/metrics/` плюс structlog-дубли. Цены input/output за 1M токенов приходят из окружения, по умолчанию 0.

**Blocked by:** 01.

**Status:** done

- [x] Декоратор реализует Protocol `InferenceProvider` и подключается в композиционном корне; слои не нарушены
- [x] Каждый вызов модели пишет `llm_call` со всеми полями задания (timestamp, run id, модель, токены, латентность, стоимость, шаг)
- [x] После завершения Запуска агента пишется `run`: шаги, суммарные токены, длительность, success
- [x] Суммы `run` сходятся с `llm_call` того же запуска (тест на скриптованном провайдере)
- [x] Хранилище append-only: записи переживают рестарт, не перезаписываются
- [x] Ни одно событие не содержит содержимого сообщений, команд и ответов модели
- [x] structlog-дубль коррелирует с JSONL по `request_id`
- [x] Все четыре гейта зелёные

## Comments

- Реализовано агентом на ветке `token-audit`. Новое: пакет `src/bot/metrics/` (`models.py` — frozen-рекорды `LlmCallRecord`/`RunRecord`; `recorder.py` — append-only JSONL `.data/metrics/events.jsonl` + structlog-дубликаты; `collector.py` — аккумулятор по `RequestId` и порт `RunMetrics`; `provider.py` — декоратор `MeteredInferenceProvider`, пишет `llm_call` и на неудачный вызов). `ApplicationService` закрывает запуск `finish_run` в `finally` (success = финальный ответ без упора в лимит); декоратор подключён в `main.py`. Цены — `METRICS_INPUT_PRICE_PER_MTOK` / `METRICS_OUTPUT_PRICE_PER_MTOK`, по умолчанию 0. Тесты: `tests/unit/metrics/` (16), сервис/настройки/wiring расширены; сквозной тест метрик в `tests/integration/test_wiring.py`. Ревью Standards/Spec: блокеров нет; P2 (необязательный `metrics` в сервисе) устранён — параметр обязателен. Гейты: ty / ruff check / ruff format --check / pytest (188 passed) — зелёные. Нюанс: giveup-повтор идёт с тем же `request_id`, поэтому оба цикла дают одну запись `run` (соответствует «run id = RequestId» тикета).
