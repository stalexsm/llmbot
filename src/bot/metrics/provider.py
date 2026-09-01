"""Декоратор на шве InferenceProvider: llm_call на каждый вызов модели.

Реализует Protocol ``InferenceProvider`` структурно и подключается в
композиционном корне поверх настоящего провайдера. Измеряет латентность
каждого вызова (включая неудачный) и передаёт её с usage-полями ответа
в ``RunMetricsCollector``; сам записью событий не занимается.
"""

import time

from bot.inference.models import InferenceRequest, InferenceResponse
from bot.inference.provider import InferenceProvider
from bot.metrics.collector import RunMetricsCollector


class MeteredInferenceProvider:
    """Замеряет и учитывает каждый вызов модели поверх внутреннего провайдера."""

    def __init__(self, inner: InferenceProvider, collector: RunMetricsCollector) -> None:
        self._inner = inner
        self._collector = collector

    async def generate(
        self,
        request: InferenceRequest,
    ) -> InferenceResponse:
        started_at = time.monotonic()
        try:
            response = await self._inner.generate(request)
        except Exception:
            # Неудачный вызов — тоже вызов модели: пишем событие без токенов
            # и пробрасываем исключение дальше, как его отдал внутренний слой.
            self._collector.record_llm_call(
                request_id=request.request_id,
                model=request.model,
                latency_ms=self._latency_ms(started_at),
                usage=None,
            )
            raise
        self._collector.record_llm_call(
            request_id=request.request_id,
            model=request.model,
            latency_ms=self._latency_ms(started_at),
            usage=response.usage,
        )
        return response

    @staticmethod
    def _latency_ms(started_at: float) -> int:
        return int((time.monotonic() - started_at) * 1000)
