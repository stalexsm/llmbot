# Telegram LLM Bot

Простой Telegram-бот на Python, который принимает текстовое сообщение пользователя,
отправляет его локальной языковой модели через [Ollama](https://ollama.com) и возвращает
ответ в тот же чат.

Первая версия намеренно минимальна, но границы слоёв рассчитаны на будущее расширение:
агент, память, инструменты и дополнительные провайдеры инференса — без изменения
Telegram-слоя.

## Архитектура

```text
Telegram (aiogram adapter)
        ↓
ApplicationService (use case: ProcessUserMessage)
        ↓
InferenceProvider (Protocol)
        ↓
OllamaInferenceProvider (adapter, httpx)
        ↓
Ollama /api/chat (non-streaming)
```

Ключевые принципы:

- слои общаются только через типизированные контракты;
- Telegram не знает про Ollama, приложение не знает про aiogram и JSON Ollama;
- никакого глобального изменяемого состояния: все объекты создаются в
  composition root (`src/bot/main.py`) и внедряются явно через конструкторы;
- семантические идентификаторы — `typing.NewType`, внутренние модели —
  неизменяемые `dataclass(frozen=True)`, роли — `StrEnum`;
- история диалога не хранится: каждое сообщение — независимый запрос;
- ошибки инфраструктуры маппятся в прикладные исключения и не попадают
  к пользователю в исходном виде.

## Требования

- Python 3.13
- [uv](https://docs.astral.sh/uv/)
- Запущенный [Ollama](https://ollama.com) с моделью:

```bash
ollama pull qwen3:1.7b
# альтернатива:
ollama pull tinyllama
```

## Конфигурация

Скопируйте `env.example` в `.env` и заполните токен бота (получите у @BotFather):

```text
TELEGRAM_BOT_TOKEN=123456:ABC-...
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:1.7b
```

Дополнительные переменные (со значениями по умолчанию):

| Переменная | По умолчанию | Описание |
|---|---|---|
| `OLLAMA_TIMEOUT_SECONDS` | `120` | таймаут запроса к Ollama |
| `TELEGRAM_TIMEOUT_SECONDS` | `30` | таймаут запросов к Telegram API |
| `LOG_LEVEL` | `INFO` | уровень логирования |
| `LOG_FORMAT` | `console` | `console` или `json` |

`.env` исключён из Git; токен хранится как `SecretStr` и никогда не логируется.

## Запуск

```bash
uv sync                 # создать venv и установить зависимости (uv.lock фиксирует граф)
uv run python -m bot.main
```

## Проверки

```bash
uv run ty check                  # проверка типов
uv run ruff check .              # линтер
uv run ruff format --check .     # форматирование
uv run pytest                    # тесты (Ollama и Telegram не требуются)
```

## Структура проекта

```text
src/bot/
├── main.py              # composition root: сборка графа зависимостей, поллинг
├── config/settings.py   # pydantic-settings: переменные окружения и .env
├── domain/
│   ├── ids.py           # семантические идентификаторы (NewType)
│   └── messages.py      # MessageRole, InferenceMessage
├── application/
│   ├── errors.py        # прикладные исключения
│   ├── models.py        # UserMessageRequest / UserMessageResponse
│   └── service.py       # ApplicationService — сценарий ProcessUserMessage
├── inference/
│   ├── models.py        # InferenceRequest / InferenceResponse
│   ├── provider.py      # Protocol InferenceProvider
│   └── ollama.py        # адаптер Ollama (httpx + Pydantic на границе JSON)
└── telegram/
    ├── handlers.py      # aiogram-обработчики
    └── mapper.py        # aiogram Message → UserMessageRequest

tests/
├── unit/                # сервис, адаптер Ollama (MockTransport), маппер, хендлеры
└── integration/         # полный цикл Telegram → приложение → Ollama (без сети)
```

## Логирование

`structlog`, настройка выполняется в composition root, логгеры внедряются через
конструкторы. В логах — только идентификаторы и метрики (`request_id`, `model`,
`duration_ms`, `status`), но не содержимое сообщений пользователя и не секреты.
