"""Unit tests for the ``ProcessUserMessage`` use case."""

from uuid import uuid4

import pytest
import structlog.stdlib
from structlog.testing import CapturingLogger

from bot.application.errors import (
    EmptyInferenceResponseError,
    InferenceTimeoutError,
    InferenceUnavailableError,
)
from bot.application.models import UserMessageRequest
from bot.application.service import ApplicationService
from bot.domain.ids import (
    ModelId,
    RequestId,
    TelegramChatId,
    TelegramMessageId,
    TelegramUserId,
)
from bot.domain.messages import InferenceMessage, MessageRole
from bot.inference.provider import InferenceProvider
from tests.fakes import FailingInferenceProvider, MockInferenceProvider


def make_request(text: str) -> UserMessageRequest:
    return UserMessageRequest(
        request_id=RequestId(str(uuid4())),
        user_id=TelegramUserId(7),
        chat_id=TelegramChatId(100),
        message_id=TelegramMessageId(42),
        text=text,
    )


def make_service(
    logger: structlog.stdlib.BoundLogger,
    provider: InferenceProvider,
    model: str = "qwen3:1.7b",
) -> ApplicationService:
    return ApplicationService(
        inference=provider,
        model=ModelId(model),
        logger=logger,
    )


async def test_returns_model_response(logger: structlog.stdlib.BoundLogger) -> None:
    provider = MockInferenceProvider(response_content="Ответ модели")
    service = make_service(logger, provider)

    response = await service.process_message(make_request("Привет"))

    assert response.text == "Ответ модели"
    assert len(provider.requests) == 1


@pytest.mark.parametrize("model", ["qwen3:1.7b", "tinyllama"])
async def test_builds_inference_request(logger: structlog.stdlib.BoundLogger, model: str) -> None:
    provider = MockInferenceProvider()
    service = make_service(logger, provider, model=model)

    request = make_request("Привет")
    await service.process_message(request)

    inference_request = provider.requests[0]
    assert inference_request.request_id == request.request_id
    assert inference_request.model == ModelId(model)
    assert inference_request.messages == (
        InferenceMessage(role=MessageRole.USER, content="Привет"),
    )


async def test_consecutive_messages_do_not_share_history(
    logger: structlog.stdlib.BoundLogger,
) -> None:
    provider = MockInferenceProvider()
    service = make_service(logger, provider)

    await service.process_message(make_request("Первое сообщение"))
    await service.process_message(make_request("Второе сообщение"))

    assert len(provider.requests) == 2
    assert provider.requests[0].messages[0].content == "Первое сообщение"
    assert provider.requests[1].messages == (
        InferenceMessage(role=MessageRole.USER, content="Второе сообщение"),
    )


async def test_empty_model_response_raises_explicitly(
    logger: structlog.stdlib.BoundLogger,
) -> None:
    provider = MockInferenceProvider(response_content="   ")
    service = make_service(logger, provider)

    with pytest.raises(EmptyInferenceResponseError):
        await service.process_message(make_request("Привет"))


@pytest.mark.parametrize(
    "error",
    [
        InferenceTimeoutError("timed out"),
        InferenceUnavailableError("unreachable"),
    ],
)
async def test_infrastructure_errors_propagate_as_application_errors(
    logger: structlog.stdlib.BoundLogger, error: Exception
) -> None:
    service = make_service(logger, FailingInferenceProvider(error))

    with pytest.raises(type(error)):
        await service.process_message(make_request("Привет"))


async def test_user_content_is_not_logged(capturing_logger: CapturingLogger) -> None:
    import structlog.stdlib

    logger = structlog.stdlib.BoundLogger(capturing_logger, [], {})
    provider = MockInferenceProvider()
    service = make_service(logger, provider)

    await service.process_message(make_request("Секретный текст пользователя"))

    logged_events = [call.kwargs.get("event") for call in capturing_logger.calls]
    assert "inference_finished" in logged_events
    for call in capturing_logger.calls:
        assert "Секретный текст пользователя" not in str(call.kwargs)
