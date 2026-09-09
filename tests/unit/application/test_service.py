"""Unit tests for the ``ProcessUserMessage`` use case around the agentic loop."""

import asyncio
from pathlib import Path
from uuid import uuid4

import pytest
import structlog.stdlib
from structlog.testing import CapturingLogger

from bot.agent.exec import ExecTool
from bot.agent.loop import AgentLoop
from bot.agent.prompts import build_date_block
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
)
from bot.domain.messages import InferenceMessage, MessageRole
from bot.inference.models import InferenceRequest, InferenceResponse
from bot.inference.provider import InferenceProvider
from bot.metrics.collector import RunMetrics
from bot.sessions.migrations import apply_migrations
from bot.sessions.store import ChatSessionStore
from tests.fakes import (
    FailingInferenceProvider,
    MockInferenceProvider,
    RecordingProgress,
    ScriptedInferenceProvider,
    SpyMetricsCollector,
    exec_call_response,
    final_response,
)

CHAT = TelegramChatId(100)
SYSTEM_PROMPT = "Ты тестовый агент."


def make_session_store(directory: Path, logger: structlog.stdlib.BoundLogger) -> ChatSessionStore:
    """Реальный SQLite-store над мигрированной временной БД."""
    database = directory / "chats.db"
    apply_migrations(database)
    return ChatSessionStore(database=database, logger=logger)


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
    metrics: RunMetrics | None = None,
) -> ApplicationService:
    if metrics is None:
        metrics = SpyMetricsCollector()
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
        sessions=make_session_store(directory, logger),
        history_limit=history_limit,
        logger=logger,
        metrics=metrics,
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
        InferenceMessage(
            role=MessageRole.SYSTEM,
            content=f"{SYSTEM_PROMPT}\n\n{build_date_block()}",
        ),
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
        InferenceMessage(
            role=MessageRole.SYSTEM,
            content=f"{SYSTEM_PROMPT}\n\n{build_date_block()}",
        ),
        InferenceMessage(role=MessageRole.USER, content="Меня зовут Саша"),
        InferenceMessage(role=MessageRole.ASSISTANT, content="Ответ модели"),
        InferenceMessage(role=MessageRole.USER, content="Как меня зовут?"),
    )


async def test_history_window_limits_request_messages(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    provider = MockInferenceProvider()
    service = make_service(logger, provider, tmp_path, history_limit=5)
    sessions = make_session_store(tmp_path, logger)
    for i in range(8):
        sessions.append(
            CHAT,
            InferenceMessage(role=MessageRole.USER, content=f"старое {i}"),
        )

    await service.process_message(make_request("новый вопрос"))

    messages = provider.requests[0].messages
    assert len(messages) == 7  # системный промпт + окно 5 + текущее сообщение
    assert [message.content for message in messages] == [
        f"{SYSTEM_PROMPT}\n\n{build_date_block()}",
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

    sessions = make_session_store(tmp_path, logger)
    assert sessions.load(CHAT) == ()


async def test_empty_model_response_raises_explicitly(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    # Цикл дважды повторяет пустой шаг; после исчерпания повторов — исключение.
    provider = ScriptedInferenceProvider([final_response("   ")] * 3)
    service = make_service(logger, provider, tmp_path)

    with pytest.raises(EmptyInferenceResponseError):
        await service.process_message(make_request("Привет"))

    sessions = make_session_store(tmp_path, logger)
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


async def test_tool_exchange_is_not_persisted_to_session(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    # В сессию попадает только диалог: tool-обмен живёт внутри одного запуска.
    provider = ScriptedInferenceProvider(
        [exec_call_response("echo привет"), final_response("Готово: привет")]
    )
    service = make_service(logger, provider, tmp_path, with_exec_tool=True)

    response = await service.process_message(make_request("покажи файлы"))

    assert response.text == "Готово: привет"
    sessions = make_session_store(tmp_path, logger)
    history = sessions.load(CHAT)
    assert [(message.role, message.content) for message in history] == [
        (MessageRole.USER, "покажи файлы"),
        (MessageRole.ASSISTANT, "Готово: привет"),
    ]
    assert all(not message.tool_calls for message in history)


async def test_giveup_run_is_not_persisted_and_retried(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    """Отказ-подобный финал после провала команды: без записи в сессию, с повтором."""
    provider = ScriptedInferenceProvider(
        [
            exec_call_response("false"),
            final_response("Не удалось получить данные: команда завершилась с ошибкой."),
            exec_call_response("echo данные"),
            final_response("Вот данные: 42"),
        ]
    )
    service = make_service(logger, provider, tmp_path, with_exec_tool=True)

    response = await service.process_message(make_request("дай данные"))

    # Принят второй запуск; бракованный не дошёл до пользователя и в сессию.
    assert response.text == "Вот данные: 42"
    sessions = make_session_store(tmp_path, logger)
    history = sessions.load(CHAT)
    assert [m.role for m in history] == [MessageRole.USER, MessageRole.ASSISTANT]
    assert history[1].content == "Вот данные: 42"
    # Всего четыре запроса модели: два на забракованный запуск, два на принятый.
    assert len(provider.requests) == 4


async def test_successful_run_is_not_retried(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    provider = ScriptedInferenceProvider([final_response("Обычный успешный ответ")])
    service = make_service(logger, provider, tmp_path)

    response = await service.process_message(make_request("привет"))

    assert response.text == "Обычный успешный ответ"
    assert len(provider.requests) == 1


async def test_giveup_without_any_tool_attempt_is_retried(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    """Модель отказалась, даже не попробовав инструменты, — повторяем запуск."""
    provider = ScriptedInferenceProvider(
        [
            final_response("Я не знаю точного ответа на этот вопрос."),
            final_response("В Турции сейчас солнечно, +28°C."),
        ]
    )
    service = make_service(logger, provider, tmp_path)

    response = await service.process_message(make_request("Какая погода в Турции?"))

    assert response.text == "В Турции сейчас солнечно, +28°C."
    assert len(provider.requests) == 2
    # Бракованный отказ в сессию не попал: там только принятый обмен.
    sessions = make_session_store(tmp_path, logger)
    history = sessions.load(CHAT)
    assert [m.content for m in history] == [
        "Какая погода в Турции?",
        "В Турции сейчас солнечно, +28°C.",
    ]


async def test_giveup_after_real_attempts_is_answered_as_is(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    """Честное «не знаю» после реальной работы с данными — не брак."""
    provider = ScriptedInferenceProvider(
        [
            exec_call_response("curl -s example.com/data"),
            final_response("Данные недоступны: сервис не отвечает."),
        ]
    )
    service = make_service(logger, provider, tmp_path, with_exec_tool=True)

    response = await service.process_message(make_request("какой курс?"))

    assert "Данные недоступны" in response.text
    # Один запуск: шаг инструмента + финал. Повтора нет — попытка была честной.
    assert len(provider.requests) == 2


async def test_giveup_after_only_reading_skill_is_retried(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    """Прочитал скилл и бросил, не дойдя до данных, — брак: повторяем."""
    provider = ScriptedInferenceProvider(
        [
            exec_call_response("cat skills/weather/SKILL.md"),
            final_response("Я не знаю точного ответа: доступ к живым данным ограничен."),
            exec_call_response('curl -s --max-time 15 "wttr.in/Turkey?0"'),
            final_response("В Турции сейчас ☀️ +28°C, ветер 10 км/ч."),
        ]
    )
    service = make_service(logger, provider, tmp_path, with_exec_tool=True)

    response = await service.process_message(make_request("Какая погода в Турции?"))

    assert response.text == "В Турции сейчас ☀️ +28°C, ветер 10 км/ч."
    assert len(provider.requests) == 4
    sessions = make_session_store(tmp_path, logger)
    history = sessions.load(CHAT)
    assert [m.content for m in history][-1] == "В Турции сейчас ☀️ +28°C, ветер 10 км/ч."


async def test_step_limit_stop_reports_flag_and_persists_user_message(
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
    # Финального ответа нет, tool-обмен не хранится — в сессии только вопрос.
    sessions = make_session_store(tmp_path, logger)
    history = sessions.load(CHAT)
    assert [message.role for message in history] == [MessageRole.USER]


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
        f"{SYSTEM_PROMPT}\n\n{build_date_block()}",
        "первый",
        "Ответ модели",
        "второй",
    ]
    sessions = make_session_store(tmp_path, logger)
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
        InferenceMessage(
            role=MessageRole.SYSTEM,
            content=f"{SYSTEM_PROMPT}\n\n{build_date_block()}",
        ),
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


async def test_successful_run_finishes_metrics_with_success(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    metrics = SpyMetricsCollector()
    service = make_service(logger, MockInferenceProvider(), tmp_path, metrics=metrics)
    request = make_request("Привет")

    await service.process_message(request)

    assert metrics.finished == [(request.request_id, True)]


async def test_step_limit_run_finishes_metrics_without_success(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    metrics = SpyMetricsCollector()
    provider = ScriptedInferenceProvider(
        [exec_call_response("echo раз"), exec_call_response("echo два")]
    )
    service = make_service(
        logger, provider, tmp_path, step_limit=2, with_exec_tool=True, metrics=metrics
    )
    request = make_request("сделай много")

    response = await service.process_message(request)

    assert response.stopped_by_step_limit is True
    assert metrics.finished == [(request.request_id, False)]


async def test_failed_run_still_finishes_metrics_without_success(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    metrics = SpyMetricsCollector()
    service = make_service(
        logger,
        FailingInferenceProvider(InferenceUnavailableError("down")),
        tmp_path,
        metrics=metrics,
    )
    request = make_request("Привет")

    with pytest.raises(InferenceUnavailableError):
        await service.process_message(request)

    assert metrics.finished == [(request.request_id, False)]
