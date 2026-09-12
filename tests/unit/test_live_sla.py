"""SLA-тест задержки: бюджет из окружения, канонический запрос, репорт."""

import pytest

from bot.domain.ids import ModelId, RequestId
from bot.inference.models import InferenceRequest, InferenceResponse, InferenceUsage
from tests.fakes import MockInferenceProvider
from tests.live.sla import (
    BUDGET_ENV_VAR,
    CANONICAL_PROMPT,
    DEFAULT_LATENCY_SECONDS,
    build_canonical_messages,
    build_sla_report,
    latency_budget_seconds,
    measure_generate,
)

_REQUEST = InferenceRequest(
    request_id=RequestId("sla-unit"),
    model=ModelId("qwen3:4b"),
    messages=build_canonical_messages(),
)


class TestLatencyBudget:
    def test_missing_env_var_falls_back_to_task_default(self) -> None:
        assert latency_budget_seconds({}) == DEFAULT_LATENCY_SECONDS

    def test_empty_value_falls_back_to_task_default(self) -> None:
        assert latency_budget_seconds({BUDGET_ENV_VAR: "  "}) == DEFAULT_LATENCY_SECONDS

    def test_reads_value_from_environ(self) -> None:
        assert latency_budget_seconds({BUDGET_ENV_VAR: "10"}) == 10.0

    def test_broken_value_is_clear_value_error(self) -> None:
        with pytest.raises(ValueError, match=BUDGET_ENV_VAR):
            latency_budget_seconds({BUDGET_ENV_VAR: "четыре"})

    def test_non_positive_budget_rejected(self) -> None:
        with pytest.raises(ValueError, match="положительн"):
            latency_budget_seconds({BUDGET_ENV_VAR: "0"})

    def test_negative_budget_rejected(self) -> None:
        with pytest.raises(ValueError, match="положительн"):
            latency_budget_seconds({BUDGET_ENV_VAR: "-1"})


class TestCanonicalMessages:
    def test_single_short_user_message(self) -> None:
        messages = build_canonical_messages()

        assert len(messages) == 1
        assert messages[0].role.value == "user"
        assert messages[0].content == CANONICAL_PROMPT

    def test_canonical_prompt_is_short(self) -> None:
        # Короткость — часть контракта канонического вызова: длинный промпт
        # раздувает обработку и делает SLA-порог зависимым от длины.
        assert len(CANONICAL_PROMPT) <= 120


class TestMeasureGenerate:
    async def test_returns_response_with_elapsed_seconds(self) -> None:
        provider = MockInferenceProvider(response_content="Париж")

        response, elapsed_seconds = await measure_generate(provider, _REQUEST)

        assert response.content == "Париж"
        assert elapsed_seconds >= 0.0
        assert provider.requests == [_REQUEST]


class TestSlaReport:
    def test_full_usage_reported_as_breakdown(self) -> None:
        response = InferenceResponse(
            request_id=_REQUEST.request_id,
            content="Париж",
            usage=InferenceUsage(
                prompt_tokens=12,
                completion_tokens=5,
                total_duration_ns=2_000_000_000,
                load_duration_ns=100_000_000,
                prompt_eval_duration_ns=456_000_000,
                eval_duration_ns=320_000_000,
            ),
        )

        report = build_sla_report(elapsed_seconds=1.2345, budget_seconds=4.0, response=response)

        assert "1.234" in report  # полный ответ
        assert "4.0" in report  # бюджет
        assert "0.456" in report  # прокси-TTFT: обработка промпта
        assert "12" in report  # токены промпта
        assert "0.100" in report  # загрузка модели
        assert "0.320" in report  # генерация
        assert "5" in report  # токены генерации

    def test_proxy_line_carries_no_assert_marker(self) -> None:
        # Прокси-TTFT репортится без ассерта: строка помечена как прокси,
        # а не как проверяемый порог.
        response = InferenceResponse(
            request_id=_REQUEST.request_id,
            content="Париж",
            usage=InferenceUsage(prompt_eval_duration_ns=456_000_000),
        )

        report = build_sla_report(elapsed_seconds=1.0, budget_seconds=4.0, response=response)

        assert "прокси" in report
        assert "0.456" in report

    def test_missing_usage_is_reported_without_crash(self) -> None:
        response = InferenceResponse(request_id=_REQUEST.request_id, content="Париж")

        report = build_sla_report(elapsed_seconds=1.0, budget_seconds=4.0, response=response)

        assert "usage" in report
