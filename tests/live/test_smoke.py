"""Живой смоук каркаса: провайдер инференса доезжает до настоящего Ollama.

Проверяет сам шов живых тестов — одиночный короткий вызов модели возвращает
непустой контент. Без работающего Ollama тест скипается (фикстура
``live_ollama``); маркер ``live`` ставится конфтестом автоматически.
"""

import structlog.stdlib

from bot.domain.ids import ModelId, RequestId
from bot.domain.messages import InferenceMessage, MessageRole
from bot.inference.models import InferenceRequest
from bot.inference.ollama import OllamaInferenceProvider
from tests.live.support import LiveOllama


async def test_live_provider_answers_non_empty(
    live_ollama: LiveOllama,
    logger: structlog.stdlib.BoundLogger,
) -> None:
    provider = OllamaInferenceProvider(
        client=live_ollama.client,
        base_url=live_ollama.base_url,
        timeout_seconds=live_ollama.timeout_seconds,
        logger=logger,
        think=live_ollama.think,
        num_ctx=live_ollama.num_ctx,
    )
    response = await provider.generate(
        InferenceRequest(
            request_id=RequestId("live-smoke"),
            model=ModelId(live_ollama.model),
            messages=(
                InferenceMessage(role=MessageRole.USER, content="Ответь ровно одним словом: жив?"),
            ),
        )
    )

    assert response.content.strip(), "живая модель вернула пустой контент"
