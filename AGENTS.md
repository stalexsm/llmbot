# llmbot

Telegram-бот с агентным циклом поверх Ollama. Слои и единственный способ их
общения — типизированные контракты:

```text
Telegram-адаптер (aiogram)
    ↓
ApplicationService (use case: ProcessUserMessage)
    ↓
AgentLoop (агентный цикл)
    ↓
InferenceProvider (Protocol)
    ↓
OllamaInferenceProvider (адаптер, httpx)
    ↓
Ollama
```

Центральный принцип: `Telegram ≠ Application ≠ Agent ≠ Inference ≠ Ollama`.

## Справочники

Читай справочник, когда задача попадает в его ветку. Прочитанный документ
остаётся в контексте до конца сессии: при повторной необходимости опирайся на
уже прочитанное, повторное чтение того же файла — лишняя работа.

- Меняешь границы слоёв, добавляешь компонент или провайдер, трогаешь
  DI/композиционный корень или обработку ошибок между слоями —
  `docs/architecture.md`.
- Пишешь или правишь Python-код (типизация, `NewType`, dataclasses, Pydantic,
  `Protocol`/`StrEnum`) — `docs/code-style.md`.
- Трогаешь конфигурацию/`Settings`, логирование, таймауты или `RequestId` —
  `docs/runtime.md`.

## Инструменты и проверки

- Менеджер пакетов — только `uv`; `uv.lock` коммитится. Зависимости и их версии 
- читаются из `pyproject.toml` + `uv.lock`, а не дублируются здесь.
- Все команды — через `uv run`:
  - `uv run python -m bot.main` — запуск бота;
  - `uv run ty check` — типы;
  - `uv run ruff check .` — линт;
  - `uv run ruff format --check .` — формат;
  - `uv run pytest` — тесты.

## Тесты

- Юнит-тесты приложения не требуют запущенный Ollama: `InferenceProvider`
  подменяется фейком (`tests/fakes.py`).
- Четыре гейта должны проходить перед завершением работы:
  `ty check`, `ruff check`, `ruff format --check`, `pytest`.

## Agent skills

### Issue tracker

Issues are tracked as local markdown files under `.scratch/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default triage labels are used as-is: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
