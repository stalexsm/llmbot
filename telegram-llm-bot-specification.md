# Telegram LLM Bot — Final Technical Specification

## 1. Purpose

Develop a simple Telegram bot in Python that accepts a user's text message, sends it to a local language model through Ollama, and returns the model response to the same Telegram chat.

The first version is intentionally small, but its boundaries must support future expansion to an Agent, memory, tools, and additional inference providers without redesigning the Telegram layer.

### Current flow

```text
Telegram
    ↓
Telegram Adapter
    ↓
Application Service
    ↓
InferenceProvider
    ↓
Ollama
    ↓
Local LLM
```

### Future flow

```text
Telegram
    ↓
Application Service
    ↓
Agent
    ↓
InferenceProvider
    ↓
Ollama / other provider
```

The Agent is a future extension point only. It is not implemented in the first version.

---

# 2. Functional requirements

## 2.1 Telegram

The bot must:

1. receive text messages from Telegram;
2. process `/start`;
3. pass the user's text to the application layer;
4. return the generated LLM response to the same chat.

The Telegram layer must not contain Ollama-specific logic.

## 2.2 LLM

Inference must be performed through a local Ollama instance.

Default model:

```text
qwen3:1.7b
```

Alternative supported model:

```text
tinyllama
```

The model is configured through environment variables.

The first version uses non-streaming inference.

## 2.3 No conversation memory

The bot must not persist conversation history.

Every message is an independent request:

```text
User Message
    ↓
Inference Request
    ↓
LLM
    ↓
Bot Reply
```

The system must not use:

- a database;
- Redis;
- conversation history;
- persistent memory;
- session context.

---

# 3. Technology stack

| Area | Technology |
|---|---|
| Language | Python 3.13 |
| Package manager | uv |
| Telegram | aiogram 3.30.0 |
| LLM runtime | Ollama |
| HTTP client | httpx |
| Configuration | pydantic-settings |
| Internal models | dataclasses |
| Semantic identifiers | typing.NewType |
| Interfaces | typing.Protocol |
| Enums | enum.StrEnum |
| Logging | structlog |
| Type checking | ty |
| Linting / formatting | Ruff |
| Tests | pytest |
| Async tests | pytest-asyncio |

`ty` is the selected type checker. It is currently a beta tool, so its exact version must be locked in `uv.lock`; CI must run the locked version rather than an unconstrained latest release.

---

# 4. Package management with uv

`uv` is the only package/dependency manager for the project.

Do not maintain a separate `requirements.txt`.

The repository must contain:

```text
pyproject.toml
uv.lock
.python-version
```

`.python-version`:

```text
3.13
```

`pyproject.toml` must declare:

```toml
requires-python = ">=3.13,<3.14"
```

All dependencies are installed and executed through `uv`.

Examples:

```bash
uv sync
uv run python -m bot.main
uv run ty check
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

`uv.lock` must be committed to Git.

---

# 5. Dependency versions

The Telegram dependency is explicitly pinned:

```toml
aiogram == 3.30.0
```

The project must use the lock file to make the complete dependency graph reproducible.

Core runtime dependencies:

```text
aiogram==3.30.0
httpx
pydantic-settings
structlog
```

Development dependencies:

```text
ty
ruff
pytest
pytest-asyncio
```

Exact resolved versions are recorded in `uv.lock`.

---

# 6. Architecture

The application is divided into explicit layers.

```text
┌─────────────────────────────┐
│       Telegram Layer        │
│          aiogram            │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│      Application Layer      │
│       use cases/services    │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│      Domain Contracts       │
│ Protocols + typed models    │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Infrastructure / Adapters   │
│           Ollama            │
└─────────────────────────────┘
```

The application layer must depend on abstractions, not on the concrete Ollama implementation.

---

# 7. Mandatory architectural rules

## 7.1 No global runtime state

Global mutable variables and global runtime instances are forbidden.

Forbidden:

```python
bot = Bot(...)
ollama = OllamaInferenceProvider(...)
application = ApplicationService(...)
settings = Settings(...)
```

as module-level runtime state.

Definitions such as classes, functions, constants, `NewType`, and `Protocol` are allowed.

---

## 7.2 Explicit dependency injection

Dependencies must be injected explicitly, preferably through constructors.

Example:

```python
class ApplicationService:
    def __init__(
        self,
        inference: InferenceProvider,
    ) -> None:
        self._inference = inference
```

A component must not create its own infrastructure dependencies.

Forbidden:

```python
class ApplicationService:
    def __init__(self) -> None:
        self._inference = OllamaInferenceProvider(...)
```

---

## 7.3 No DI container in the first version

Do not use Dishka in the first version.

Dependency injection is performed manually in the composition root.

A DI container may be introduced later if the dependency graph becomes substantially more complex, for example after adding Agent, Memory, Tools, repositories, multiple providers, and multiple scopes.

Introducing a DI container must not require changes to domain/application contracts.

---

## 7.4 Composition root

Runtime objects are created in one composition root, e.g. `main.py`.

Conceptually:

```text
main()
 ├── load Settings
 ├── create HTTP client
 ├── create OllamaInferenceProvider
 ├── create ApplicationService
 ├── create Telegram Bot
 ├── register handlers
 └── start polling
```

Lifecycle ownership must be explicit. Resources such as the HTTP client and Telegram bot must be closed correctly.

---

# 8. Strong typing

The project must use strict static typing.

All public functions and methods must have explicit type annotations.

Avoid:

```python
Any
dict[str, Any]
cast(...)
```

as a way to bypass type errors.

`Any` may only be used at a justified external/untyped boundary and must not leak into domain/application contracts.

The project must pass:

```bash
uv run ty check
```

without type errors.

---

# 9. Semantic identifiers with NewType

Every semantically different identifier must use its own `NewType`.

Example:

```python
from typing import NewType

TelegramUserId = NewType("TelegramUserId", int)
TelegramChatId = NewType("TelegramChatId", int)
TelegramMessageId = NewType("TelegramMessageId", int)

RequestId = NewType("RequestId", str)
ModelId = NewType("ModelId", str)
```

Do not use plain `int` or `str` for semantically distinct IDs inside application/domain contracts.

For example:

```python
def process_user(user_id: TelegramUserId) -> None: ...
```

must not accept `TelegramChatId` without an explicit conversion.

Future identifiers such as `AgentId` and `ToolId` must also receive their own `NewType`.

---

# 10. Internal models

Internal domain/application schemas must use dataclasses.

Prefer immutable models:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceMessage:
    role: MessageRole
    content: str
```

```python
@dataclass(frozen=True)
class InferenceRequest:
    request_id: RequestId
    model: ModelId
    messages: tuple[InferenceMessage, ...]
```

```python
@dataclass(frozen=True)
class InferenceResponse:
    request_id: RequestId
    content: str
```

Raw dictionaries must not be used as internal application protocols.

---

# 11. Pydantic and dataclasses

Pydantic is used for external/configuration boundaries.

`pydantic-settings` is used for environment configuration.

Dataclasses are used for internal application/domain models.

Conceptually:

```text
.env / environment
        ↓
pydantic-settings
        ↓
typed configuration
        ↓
application/domain dataclasses
```

If external JSON validation requires Pydantic models, they must be converted to internal dataclasses at the boundary.

Pydantic models must not replace the required internal dataclass structures.

---

# 12. Configuration

Configuration is loaded from environment variables and `.env`.

Required:

```text
TELEGRAM_BOT_TOKEN
```

Optional/defaulted:

```text
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:1.7b
```

Example settings:

```python
class Settings(BaseSettings):
    telegram_bot_token: str
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:1.7b"
```

Configuration must be validated at startup.

The application must fail fast if required configuration is missing or invalid.

The Telegram token must never be logged.

`.env` must be listed in `.gitignore`.

---

# 13. InferenceProvider

Inference must be represented by an abstraction independent of Ollama.

Example:

```python
class InferenceProvider(Protocol):
    async def generate(
        self,
        request: InferenceRequest,
    ) -> InferenceResponse: ...
```

The application layer depends only on this protocol.

The concrete implementation is:

```text
OllamaInferenceProvider
```

Future implementations may include other local or remote inference providers without changing the application layer.

---

# 14. Ollama adapter

`OllamaInferenceProvider` is an infrastructure adapter.

Responsibilities:

- HTTP communication with Ollama;
- request serialization;
- response parsing;
- response validation;
- timeout handling;
- mapping infrastructure errors to application-level errors.

It must not contain Telegram-specific logic.

The provider must use an injected `httpx.AsyncClient`.

It must not create a new HTTP client for every inference request.

---

# 15. Ollama API

The initial implementation uses Ollama's chat API.

Conceptually:

```text
POST /api/chat
```

Example request:

```json
{
  "model": "qwen3:1.7b",
  "messages": [
    {
      "role": "user",
      "content": "Привет"
    }
  ],
  "stream": false
}
```

`stream` must be `false` in the first version.

Streaming is outside the current scope.

---

# 16. Message roles

Use a typed enum:

```python
from enum import StrEnum


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
```

The first version only needs `USER` input.

The model should nevertheless use a representation capable of supporting future assistant/system messages.

---

# 17. Application Service

The application layer exposes the use case:

```text
ProcessUserMessage
```

Example:

```python
class ApplicationService:
    async def process_message(
        self,
        request: UserMessageRequest,
    ) -> UserMessageResponse: ...
```

Responsibilities:

1. accept typed application input;
2. create an `InferenceRequest`;
3. call `InferenceProvider`;
4. map the result to an application response;
5. return the result.

The service must not know about Telegram API objects or Ollama JSON.

---

# 18. Telegram adapter

The Telegram layer is an adapter around aiogram.

Responsibilities:

- receive Telegram updates;
- extract Telegram IDs;
- map Telegram data to application DTOs;
- invoke `ApplicationService`;
- map application responses to Telegram replies;
- handle Telegram-specific errors.

It must not:

- call Ollama directly;
- construct `OllamaInferenceProvider`;
- know Ollama URLs;
- know Ollama JSON format;
- implement inference logic.

---

# 19. Request ID

Every processed user message must receive a unique `RequestId`.

Example:

```python
request_id = RequestId(str(uuid4()))
```

The request ID is used for:

- logging;
- correlation;
- troubleshooting;
- request/response tracing.

It must not be exposed to the user unless explicitly needed.

---

# 20. Logging

Use `structlog` for application logging.

Logging must be structured rather than based on manually formatted strings.

Important fields include:

```text
event
request_id
component
status
duration_ms
model
error
```

Example conceptual event:

```json
{
  "event": "inference_finished",
  "request_id": "01...",
  "model": "qwen3:1.7b",
  "duration_ms": 1832,
  "status": "success"
}
```

Do not log by default:

- Telegram bot token;
- authorization headers;
- API keys;
- complete user messages;
- complete LLM responses;
- other secrets.

User prompt/response content should not be logged by default.

---

# 21. Logger dependency

The project must not introduce global runtime logger instances.

Runtime logging dependencies must be configured in the composition root and injected into components that need them.

A logging context must not depend on hidden mutable global state.

`RequestId` should be passed explicitly where it is part of the operation's data.

---

# 22. Error handling

Infrastructure errors must not leak into the application layer.

Define application-level exceptions, for example:

```python
class InferenceError(Exception):
    pass


class InferenceTimeoutError(InferenceError):
    pass


class InferenceUnavailableError(InferenceError):
    pass
```

The Ollama adapter maps HTTP/network/timeout failures to these application errors.

The Telegram layer converts application errors into safe user-facing messages.

The user must not receive raw exception messages, stack traces, URLs, or internal infrastructure details.

---

# 23. Timeouts

All external network operations must have explicit timeouts.

At minimum:

```text
Telegram API
Ollama HTTP API
```

Timeout configuration belongs in `Settings`.

---

# 24. Future Agent

The Agent is not implemented in the first version.

The architecture must allow:

```text
Current:

Application
    ↓
InferenceProvider


Future:

Application
    ↓
Agent
    ↓
InferenceProvider
```

The Agent must be a separate abstraction boundary.

The existing Telegram layer must not need to change when Agent is introduced.

---

# 25. Agent must not use subprocess by default

`subprocess` is not part of the first version.

When an Agent is eventually implemented, its execution/isolation model must be decided separately based on its actual capabilities.

Potential future approaches include:

```text
in-process library
dedicated service
container
sandbox
worker
```

If the Agent eventually receives tool or command execution capabilities, security boundaries must be explicitly designed.

At minimum:

- least privilege;
- restricted filesystem access;
- restricted network access;
- resource limits;
- timeouts;
- explicit tool allowlist;
- audit logging;
- no unrestricted shell execution by default.

---

# 26. Future extensibility

The architecture should permit adding:

```text
Agent
Memory
Tools
Multiple inference providers
Repositories
Persistent storage
```

without coupling these concerns to Telegram.

Examples:

```text
Add Agent
→ Telegram layer remains unchanged.

Add another inference provider
→ Application layer remains unchanged.

Replace Ollama
→ Application and Telegram layers remain unchanged.

Add Memory
→ Inference abstraction remains unchanged.

Add Tools
→ Telegram layer remains unchanged.
```

Future extensibility must be achieved through stable contracts, not speculative implementations.

---

# 27. Project structure

Recommended structure:

```text
telegram-llm-bot/
│
├── .env
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
├── README.md
│
├── src/
│   └── bot/
│       ├── __init__.py
│       ├── main.py
│       │
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py
│       │
│       ├── domain/
│       │   ├── __init__.py
│       │   ├── ids.py
│       │   └── messages.py
│       │
│       ├── application/
│       │   ├── __init__.py
│       │   ├── models.py
│       │   └── service.py
│       │
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── models.py
│       │   ├── provider.py
│       │   └── ollama.py
│       │
│       └── telegram/
│           ├── __init__.py
│           ├── handlers.py
│           └── mapper.py
│
└── tests/
    ├── unit/
    │   ├── application/
    │   └── inference/
    │
    └── integration/
```

The exact module decomposition may be adjusted during implementation if it preserves the architectural boundaries.

---

# 28. Testing

Tests must not require a running Ollama instance for application unit tests.

A test implementation of `InferenceProvider` must be available:

```python
class MockInferenceProvider: ...
```

Application tests should verify:

```text
UserMessageRequest
        ↓
ApplicationService
        ↓
InferenceRequest
        ↓
MockInferenceProvider
        ↓
InferenceResponse
        ↓
UserMessageResponse
```

---

# 29. Required test scenarios

At minimum:

### `/start`

The bot responds correctly.

### Normal text

```text
"Привет"
```

is sent to inference and the generated response is returned.

### Independent requests

Two consecutive messages must not share conversation history.

### Ollama unavailable

The application returns a safe error response.

### Ollama timeout

The application handles timeout without crashing the bot.

### Invalid Ollama response

The adapter rejects malformed responses safely.

### Empty model response

The application handles an empty response explicitly.

### Request ID

Every processed message receives a unique `RequestId`.

### Type checking

```bash
uv run ty check
```

passes.

### Linting

```bash
uv run ruff check .
```

passes.

### Formatting

```bash
uv run ruff format --check .
```

passes.

### Tests

```bash
uv run pytest
```

passes.

---

# 30. Security requirements

The project must:

- keep `TELEGRAM_BOT_TOKEN` outside source code;
- load secrets from environment / `.env`;
- exclude `.env` from Git;
- never log secrets;
- never expose infrastructure errors to Telegram users;
- use network timeouts;
- avoid unrestricted command execution;
- avoid unnecessary filesystem access;
- avoid storing user messages in the first version.

---

# 31. Git requirements

`.gitignore` must include at least:

```text
.env
.venv/
__pycache__/
.pytest_cache/
.ruff_cache/
.ty/
```

`uv.lock` must be committed.

The `.env` file must never be committed.

---

# 32. Definition of Done

The first version is complete when all of the following are true:

- [ ] Python 3.13 is used.
- [ ] `uv` manages the project and dependencies.
- [ ] `aiogram==3.30.0` is used.
- [ ] Ollama is used for local inference.
- [ ] `qwen3:1.7b` works.
- [ ] `tinyllama` can be selected through configuration.
- [ ] `TELEGRAM_BOT_TOKEN` is loaded from environment / `.env`.
- [ ] No secrets are hardcoded.
- [ ] `.env` is ignored by Git.
- [ ] No mutable global runtime state exists.
- [ ] Runtime dependencies are constructed in the composition root.
- [ ] Dependencies are injected explicitly.
- [ ] Dishka is not required.
- [ ] All semantic IDs use `NewType`.
- [ ] Internal application/domain models use dataclasses.
- [ ] Internal dataclasses are immutable where practical.
- [ ] `InferenceProvider` is a `Protocol`.
- [ ] Ollama is implemented as an adapter.
- [ ] Telegram does not depend on Ollama.
- [ ] Application does not depend on concrete Ollama implementation.
- [ ] HTTP client is injected and reused.
- [ ] External operations have timeouts.
- [ ] Request IDs are generated and propagated.
- [ ] `structlog` is used for structured logging.
- [ ] User content is not logged by default.
- [ ] Infrastructure exceptions are mapped to application errors.
- [ ] No conversation history is stored.
- [ ] No Agent is implemented.
- [ ] No `subprocess` is used.
- [ ] Architecture supports a future Agent.
- [ ] Architecture supports future inference providers.
- [ ] Unit tests do not require Ollama.
- [ ] `ty check` passes.
- [ ] `ruff check` passes.
- [ ] `ruff format --check` passes.
- [ ] `pytest` passes.
- [ ] `uv.lock` is committed.

---

# 33. Architectural summary

The first version must remain intentionally small:

```text
                         Telegram
                            │
                            ▼
                    ┌───────────────┐
                    │    aiogram    │
                    │    Adapter    │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  Application  │
                    │    Service    │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Inference     │
                    │   Provider    │
                    │   Protocol    │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │     Ollama    │
                    │    Adapter    │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ qwen3:1.7b /  │
                    │   tinyllama   │
                    └───────────────┘
```

The central architectural principle is:

```text
Telegram ≠ Application ≠ Agent (future) ≠ Inference ≠ Ollama
```

Each concern has an explicit boundary and communicates through typed contracts.

The architecture should be extended only when a real requirement appears. Future Agent functionality is anticipated through the `InferenceProvider` boundary, but Agent execution, memory, tools, persistence, and process isolation are deliberately outside the first implementation.
