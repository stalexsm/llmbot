"""Судья качества: контракт JSON-вердикта и шов вызова модели-судьи."""

import pytest

from bot.domain.ids import ModelId, RequestId
from tests.fakes import MockInferenceProvider
from tests.live.judge import (
    CRITERIA,
    MIN_TIMEOUT_SECONDS,
    build_judge_messages,
    judge_answer,
    judge_timeout_seconds,
    parse_verdict,
)

_QUESTION = "Что такое Git?"
_ANSWER = "Git — система контроля версий."


class TestParseVerdict:
    def test_json_object_reads_three_scores(self) -> None:
        verdict = parse_verdict('{"politeness": 0.9, "accuracy": 0.8, "conciseness": 0.7}')

        assert (verdict.politeness, verdict.accuracy, verdict.conciseness) == (0.9, 0.8, 0.7)
        assert verdict.average == pytest.approx(0.8)

    def test_markdown_fence_is_tolerated(self) -> None:
        fenced = '```json\n{"politeness": 1, "accuracy": 0.5, "conciseness": 0}\n```'

        verdict = parse_verdict(fenced)

        assert (verdict.politeness, verdict.accuracy, verdict.conciseness) == (1.0, 0.5, 0.0)

    def test_broken_json_is_clear_value_error(self) -> None:
        with pytest.raises(ValueError, match="вердикт") as exc_info:
            parse_verdict("Судья задумался и ответил прозой.")

        # Понятная ошибка теста: в сообщении есть фрагмент сырого ответа.
        assert "Судья задумался" in str(exc_info.value)

    def test_missing_criterion_is_named_in_error(self) -> None:
        with pytest.raises(ValueError, match="accuracy"):
            parse_verdict('{"politeness": 0.9, "conciseness": 0.7}')

    def test_out_of_range_score_rejected(self) -> None:
        with pytest.raises(ValueError, match="politeness"):
            parse_verdict('{"politeness": 1.5, "accuracy": 0.8, "conciseness": 0.7}')

    def test_non_numeric_score_rejected(self) -> None:
        with pytest.raises(ValueError, match="accuracy"):
            parse_verdict('{"politeness": 0.9, "accuracy": "высоко", "conciseness": 0.7}')

    def test_non_object_verdict_rejected(self) -> None:
        with pytest.raises(ValueError, match="вердикт"):
            parse_verdict('["politeness", "accuracy", "conciseness"]')


class TestJudgeMessages:
    def test_system_prompt_requires_json_verdict(self) -> None:
        system, _ = build_judge_messages(_QUESTION, _ANSWER)

        assert system.role.value == "system"
        assert "JSON" in system.content
        for criterion in CRITERIA:
            assert criterion in system.content
        # Шкала оговорена промптом: границы 0 и 1.
        assert "0" in system.content and "1" in system.content

    def test_question_and_answer_land_in_user_message(self) -> None:
        system, user = build_judge_messages(_QUESTION, _ANSWER)

        assert user.role.value == "user"
        assert _QUESTION in user.content
        assert _ANSWER in user.content
        # Вопрос и ответ не подмешиваются в правила судьи.
        assert _QUESTION not in system.content


class TestJudgeAnswer:
    async def test_verdict_parsed_from_provider_content(self) -> None:
        provider = MockInferenceProvider(
            response_content='{"politeness": 0.9, "accuracy": 1.0, "conciseness": 0.8}'
        )

        verdict = await judge_answer(
            provider=provider,
            model=ModelId("judge:8b"),
            question=_QUESTION,
            answer=_ANSWER,
            request_id=RequestId("judge-q1"),
        )

        assert verdict.average == pytest.approx(0.9)
        [request] = provider.requests
        assert request.request_id == RequestId("judge-q1")
        assert request.model == ModelId("judge:8b")

    async def test_broken_verdict_is_value_error_not_traceback(self) -> None:
        provider = MockInferenceProvider(response_content="прозой без JSON")

        with pytest.raises(ValueError, match="вердикт"):
            await judge_answer(
                provider=provider,
                model=ModelId("judge:8b"),
                question=_QUESTION,
                answer=_ANSWER,
                request_id=RequestId("judge-q1"),
            )


class TestJudgeTimeout:
    def test_floor_raises_short_configured_timeout(self) -> None:
        assert judge_timeout_seconds(120.0) == MIN_TIMEOUT_SECONDS

    def test_configured_timeout_above_floor_is_kept(self) -> None:
        assert judge_timeout_seconds(600.0) == 600.0
