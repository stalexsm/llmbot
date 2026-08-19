"""Provider-agnostic inference contract."""

from typing import Protocol

from bot.inference.models import InferenceRequest, InferenceResponse


class InferenceProvider(Protocol):
    """Abstraction over any inference backend (Ollama today, others later)."""

    async def generate(
        self,
        request: InferenceRequest,
    ) -> InferenceResponse: ...
