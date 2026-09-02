"""Юнит-тесты учёта токенов бенчмарка (декоратор на шве провайдера)."""

from uuid import uuid4

from bot.benchmark.tracker import TokenTrackingProvider
from bot.domain.ids import ModelId, RequestId
from bot.inference.models import InferenceRequest, InferenceUsage
from tests.fakes import MockInferenceProvider


def make_request() -> InferenceRequest:
    request_id = RequestId(str(uuid4()))
    return InferenceRequest(
        request_id=request_id,
        model=ModelId("qwen3:1.7b"),
        messages=(),
    )


async def test_totals_accumulate_usage_across_calls() -> None:
    request = make_request()
    provider = TokenTrackingProvider(
        inner=MockInferenceProvider(usage=InferenceUsage(prompt_tokens=10, completion_tokens=5))
    )

    await provider.generate(request)
    await provider.generate(request)

    assert provider.totals(request.request_id) == (20, 10)


async def test_totals_are_none_without_usage() -> None:
    request = make_request()
    provider = TokenTrackingProvider(inner=MockInferenceProvider(usage=None))

    await provider.generate(request)

    assert provider.totals(request.request_id) == (None, None)


async def test_totals_of_unknown_request_are_none() -> None:
    provider = TokenTrackingProvider(inner=MockInferenceProvider())
    assert provider.totals(RequestId("unknown")) == (None, None)


async def test_generate_returns_inner_response_unchanged() -> None:
    request = make_request()
    inner = MockInferenceProvider(
        response_content="ответ", usage=InferenceUsage(prompt_tokens=7, completion_tokens=3)
    )
    provider = TokenTrackingProvider(inner=inner)

    response = await provider.generate(request)

    assert response.content == "ответ"
    assert response.usage == InferenceUsage(prompt_tokens=7, completion_tokens=3)
    # Запрос дошёл до внутреннего провайдера без изменений.
    assert inner.requests == [request]
