"""Test doubles and factories shared across unit and integration tests."""

from aiogram.types import Message

from bot.inference.models import InferenceRequest, InferenceResponse


class MockInferenceProvider:
    """Deterministic in-memory inference provider for tests.

    Structurally implements ``bot.inference.provider.InferenceProvider``.
    """

    def __init__(self, response_content: str = "Ответ модели") -> None:
        self.response_content = response_content
        self.requests: list[InferenceRequest] = []

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        self.requests.append(request)
        return InferenceResponse(request_id=request.request_id, content=self.response_content)


class FailingInferenceProvider:
    """Inference provider that always raises the configured exception."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        raise self.error


def make_telegram_message(text: str, *, message_id: int = 42, chat_id: int = 100) -> Message:
    """Build a realistic private-chat aiogram Message without any I/O."""
    return Message.model_validate(
        {
            "message_id": message_id,
            "date": 1735689600,
            "chat": {"id": chat_id, "type": "private"},
            "from": {"id": 7, "is_bot": False, "first_name": "Tester"},
            "text": text,
        }
    )
