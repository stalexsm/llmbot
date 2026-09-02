"""Учёт токенов бенчмарка: декоратор на шве InferenceProvider.

Бенчмарку нужны input/output токены каждой задачи, но не запись в JSONL:
декоратор копит usage по ``request_id`` в памяти. Живой прогон при этом
дополнительно заворачивается штатным ``MeteredInferenceProvider`` — метрики
прогона попадают в общий отчёт.
"""

from bot.domain.ids import RequestId
from bot.inference.models import InferenceRequest, InferenceResponse
from bot.inference.provider import InferenceProvider


class TokenTrackingProvider:
    """Реализует Protocol InferenceProvider и копит usage по запускам."""

    def __init__(self, inner: InferenceProvider) -> None:
        self._inner = inner
        self._prompt_tokens: dict[RequestId, int] = {}
        self._completion_tokens: dict[RequestId, int] = {}
        # Был ли хоть один вызов с usage: без счётчиков от провайдера
        # токены честно неизвестны (None), а не нулевые.
        self._usage_seen: dict[RequestId, bool] = {}

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        response = await self._inner.generate(request)
        usage = response.usage
        if usage is not None:
            self._usage_seen[request.request_id] = True
            self._prompt_tokens[request.request_id] = self._prompt_tokens.get(
                request.request_id, 0
            ) + (usage.prompt_tokens or 0)
            self._completion_tokens[request.request_id] = self._completion_tokens.get(
                request.request_id, 0
            ) + (usage.completion_tokens or 0)
        return response

    def totals(self, request_id: RequestId) -> tuple[int | None, int | None]:
        """Суммарные (prompt, completion) токены запуска; None — usage не пришёл."""
        if not self._usage_seen.get(request_id, False):
            return (None, None)
        return (
            self._prompt_tokens.get(request_id, 0),
            self._completion_tokens.get(request_id, 0),
        )
