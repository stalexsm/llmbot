"""Provider-agnostic embedding contract.

Зеркало контракта инференса: приложение зависит только от протокола,
конкретный бэкенд подключается адаптером в композиционном корне.
"""

from dataclasses import dataclass
from typing import Protocol

from bot.domain.ids import ModelId, RequestId


@dataclass(frozen=True)
class EmbeddingRequest:
    """A single non-streaming embedding request for a batch of texts."""

    request_id: RequestId
    model: ModelId
    texts: tuple[str, ...]


@dataclass(frozen=True)
class EmbeddingResponse:
    """A completed embedding result: one vector per input text, in order."""

    request_id: RequestId
    embeddings: tuple[tuple[float, ...], ...]


class EmbeddingProvider(Protocol):
    """Abstraction over any embedding backend (Ollama today, others later)."""

    async def embed(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingResponse: ...
