"""Unit tests for the ``ProcessUserMessage`` use case around the agentic loop."""

import asyncio
from pathlib import Path
from uuid import uuid4

import pytest
import structlog.stdlib
from structlog.testing import CapturingLogger

from bot.agent.exec import ExecTool
from bot.agent.loop import AgentLoop
from bot.application.errors import (
    EmptyInferenceResponseError,
    InferenceTimeoutError,
    InferenceUnavailableError,
)
from bot.application.models import UserMessageRequest, UserMessageResponse
from bot.application.service import ApplicationService
from bot.domain.ids import (
    ModelId,
    RequestId,
    TelegramChatId,
    TelegramMessageId,
    TelegramUserId,
    ToolId,
)
from bot.domain.messages import InferenceMessage, MessageRole
from bot.domain.tools import ToolCall
from bot.inference.models import InferenceRequest, InferenceResponse
from bot.inference.provider import InferenceProvider
from bot.sessions.store import ChatSessionStore
from tests.fakes import (
    FailingInferenceProvider,
    MockInferenceProvider,
    ScriptedInferenceProvider,
    exec_call_response,
    final_response,
)

CHAT = TelegramChatId(100)
SYSTEM_PROMPT = "Ты тестовый агент."


class RecordingProgress:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, bool | None]] = []

    async def command_started(self, command: str) -> None:
        self.events.append(("started", command, None))

    async def command_finished(self, command: str, succeeded: bool) -> None:
        self.events.append(("finished", command, succeeded))


def make_request(text: str) -> UserMessageRequest:
    return UserMessageRequest(
        request_id=RequestId(str(uuid4())),
        user_id=TelegramUserId(7),
        chat_id=CHAT,
        message_id=TelegramMessageId(42),
        text=text,
    )


def make_service(
    logger: structlog.stdlib.BoundLogger,
    provider: InferenceProvider,
    directory: Path,
    *,
    model: str = "qwen3:1.7b",
    history_limit: int = 20,
    step_limit: int = 10,
    with_exec_tool: bool = False,
) -> ApplicationService:
    tools: tuple[ExecTool, ...] = ()
    if with_exec_tool:
        tools = (
            ExecTool(
                cwd=directory,
                timeout_seconds=5.0,
                max_output_chars=4000,
                logger=logger,
            ),
        )
    loop = AgentLoop(
        inference=provider,
        model=ModelId(model),
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
        step_limit=step_limit,
        logger=logger,
    )
    return ApplicationService(
        agent=loop,
        sessions=ChatSessionStore(directory=directory, logger=logger),
        history_limit=history_limit,
        logger=logger,
    )


async def test_returns_model_response(logger: structlog.stdlib.BoundLogger, tmp_path: Path) -> None:
    provider = MockInferenceProvider(response_content="Ответ модели")
    service = make_service(logger, provider, tmp_path)

    response = await service.process_message(make_request("Привет"))

    assert response.text == "Ответ модели"
    assert response.stopped_by_step_limit is False
    assert len(provider.requests) == 1


@pytest.mark.parametrize("model", ["qwen3:1.7b", "tinyllama"])
async def test_builds_inference_request(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path, model: str
) -> None:
    provider = MockInferenceProvider()
    service = make_service(logger, provider, tmp_path, model=model)

    request = make_request("Привет")
    await service.process_message(request)

    inference_request = provider.requests[0]
    assert inference_request.request_id == request.request_id
    assert inference_request.model == ModelId(model)
    assert inference_request.messages == (
        InferenceMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        InferenceMessage(role=MessageRole.USER, content="Привет"),
    )


async def test_second_message_sees_first_exchange(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    provider = MockInferenceProvider()
    service = make_service(logger, provider, tmp_path)

    await service.process_message(make_request("Меня зовут Саша"))
    await service.process_message(make_request("Как меня зовут?"))

    assert len(provider.requests) == 2
    assert provider.requests[1].messages == (
        InferenceMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        InferenceMessage(role=MessageRole.USER, content="Меня зовут Саша"),
        InferenceMessage(role=MessageRole.ASSISTANT, content="Ответ модели"),
        InferenceMessage(role=MessageRole.USER, content="Как меня зовут?"),
    )


async def test_history_window_limits_request_messages(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    provider = MockInferenceProvider()
    service = make_service(logger, provider, tmp_path, history_limit=5)
    sessions = ChatSessionStore(directory=tmp_path, logger=logger)
    for i in range(8):
        sessions.append(
            CHAT,
            InferenceMessage(role=MessageRole.USER, content=f"старое {i}"),
        )

    await service.process_message(make_request("новый вопрос"))

    messages = provider.requests[0].messages
    assert len(messages) == 7  # системный промпт + окно 5 + текущее сообщение
    assert [message.content for message in messages] == [
        SYSTEM_PROMPT,
        "старое 3",
        "старое 4",
        "старое 5",
        "старое 6",
        "старое 7",
        "новый вопрос",
    ]


async def test_failed_inference_leaves_session_untouched(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    provider = FailingInferenceProvider(InferenceUnavailableError("down"))
    service = make_service(logger, provider, tmp_path)

    with pytest.raises(InferenceUnavailableError):
        await service.process_message(make_request("Привет"))

    sessions = ChatSessionStore(directory=tmp_path, logger=logger)
    assert sessions.load(CHAT) == ()


async def test_empty_model_response_raises_explicitly(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    provider = ScriptedInferenceProvider([final_response("   ")])
    service = make_service(logger, provider, tmp_path)

    with pytest.raises(EmptyInferenceResponseError):
        await service.process_message(make_request("Привет"))

    sessions = ChatSessionStore(directory=tmp_path, logger=logger)
    assert sessions.load(CHAT) == ()


@pytest.mark.parametrize(
    "error",
    [
        InferenceTimeoutError("timed out"),
        InferenceUnavailableError("unreachable"),
    ],
)
async def test_infrastructure_errors_propagate_as_application_errors(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path, error: Exception
) -> None:
    service = make_service(logger, FailingInferenceProvider(error), tmp_path)

    with pytest.raises(type(error)):
        await service.process_message(make_request("Привет"))


async def test_tool_pairs_persisted_to_session_as_linked_chain(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    provider = ScriptedInferenceProvider(
        [exec_call_response("echo привет"), final_response("Готово: привет")]
    )
    service = make_service(logger, provider, tmp_path, with_exec_tool=True)

    response = await service.process_message(make_request("покажи файлы"))

    assert response.text == "Готово: привет"
    sessions = ChatSessionStore(directory=tmp_path, logger=logger)
    history = sessions.load(CHAT)
    roles = [message.role for message in history]
    assert roles == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
        MessageRole.ASSISTANT,
    ]
    assert history[1].tool_calls == (
        ToolCall(name=ToolId("exec"), arguments='{"command": "echo привет"}'),
    )
    assert history[2].tool_name == ToolId("exec")
    assert "привет" in history[2].content


async def test_step_limit_stop_reports_flag_and_persists_exchange(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    provider = ScriptedInferenceProvider(
        [exec_call_response("echo раз"), exec_call_response("echo два")]
    )
    service = make_service(logger, provider, tmp_path, step_limit=2, with_exec_tool=True)

    response = await service.process_message(make_request("сделай много"))

    assert response.stopped_by_step_limit is True
    assert response.text == ""
    assert isinstance(response, UserMessageResponse)
    # Обмен с выполненными командами остаётся в сессии.
    sessions = ChatSessionStore(directory=tmp_path, logger=logger)
    history = sessions.load(CHAT)
    assert [message.role for message in history] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
    ]


async def test_progress_is_passed_through_to_tools(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    provider = ScriptedInferenceProvider([exec_call_response("echo шаг"), final_response("Ответ")])
    service = make_service(logger, provider, tmp_path, with_exec_tool=True)
    progress = RecordingProgress()

    await service.process_message(make_request("сделай"), progress)

    assert progress.events == [
        ("started", "echo шаг", None),
        ("finished", "echo шаг", True),
    ]


async def test_concurrent_messages_in_one_chat_are_serialized(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    gate = asyncio.Event()

    class GatedProvider:
        """Первый вызов ждёт открытия шлюза, удерживая блокировку чата."""

        def __init__(self) -> None:
            self.requests: list[InferenceRequest] = []
            self._first = True

        async def generate(self, request: InferenceRequest) -> InferenceResponse:
            self.requests.append(request)
            if self._first:
                self._first = False
                await gate.wait()
            return InferenceResponse(request_id=request.request_id, content="Ответ модели")

    provider = GatedProvider()
    service = make_service(logger, provider, tmp_path)  # type: ignore[arg-type]

    first = asyncio.create_task(service.process_message(make_request("первый")))
    await asyncio.sleep(0.01)  # первый вошёл в инференс и держит блокировку
    second = asyncio.create_task(service.process_message(make_request("второй")))
    await asyncio.sleep(0.01)  # второй встал в очередь на блокировке
    gate.set()
    await asyncio.gather(first, second)

    # Второй запрос увидел обмен первого: обработка была строго последовательной.
    assert [message.content for message in provider.requests[1].messages] == [
        SYSTEM_PROMPT,
        "первый",
        "Ответ модели",
        "второй",
    ]
    sessions = ChatSessionStore(directory=tmp_path, logger=logger)
    assert len(sessions.load(CHAT)) == 4


async def test_reset_session_clears_history(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    provider = MockInferenceProvider()
    service = make_service(logger, provider, tmp_path)
    await service.process_message(make_request("Меня зовут Саша"))

    await service.reset_session(CHAT)
    await service.process_message(make_request("Как меня зовут?"))

    assert provider.requests[1].messages == (
        InferenceMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        InferenceMessage(role=MessageRole.USER, content="Как меня зовут?"),
    )


async def test_user_content_is_not_logged(
    tmp_path: Path, capturing_logger: CapturingLogger
) -> None:
    import structlog.stdlib

    logger = structlog.stdlib.BoundLogger(capturing_logger, [], {})
    provider = MockInferenceProvider()
    service = make_service(logger, provider, tmp_path)

    await service.process_message(make_request("Секретный текст пользователя"))

    logged_events = [call.kwargs.get("event") for call in capturing_logger.calls]
    assert "agent_run_finished" in logged_events
    for call in capturing_logger.calls:
        assert "Секретный текст пользователя" not in str(call.kwargs)
