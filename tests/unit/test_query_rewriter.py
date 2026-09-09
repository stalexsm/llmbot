"""Unit-тесты LlmQueryRewriter: порт rag-слоя поверх провайдера инференса.

Дополнительный LLM-вызов переписывания обязан попадать в существующие метрики
(MeteredInferenceProvider пишет llm_call) и не оставлять содержимого запроса,
реплик диалога и переписанного текста ни в метриках, ни в логах.
"""

import json
from pathlib import Path

import structlog.stdlib
from structlog.testing import CapturingLogger

from bot.domain.ids import ModelId, RequestId
from bot.inference.models import InferenceUsage
from bot.main import LlmQueryRewriter
from bot.metrics.collector import RunMetricsCollector
from bot.metrics.provider import MeteredInferenceProvider
from bot.metrics.recorder import MetricsRecorder
from tests.fakes import MockInferenceProvider

REQUEST_ID = RequestId("rewrite-test")
TURNS = (
    "Пользователь: Сколько дней отпуска?",
    "Ассистент: 28 календарных дней.",
)


def make_rewriter(
    logger: structlog.stdlib.BoundLogger,
    tmp_path: Path,
    provider: MockInferenceProvider,
) -> LlmQueryRewriter:
    """Переписывание поверх учёта токенов — как в композиционном корне."""
    collector = RunMetricsCollector(
        recorder=MetricsRecorder(directory=tmp_path / "metrics", logger=logger),
        input_price_per_mtok=0.0,
        output_price_per_mtok=0.0,
    )
    metered = MeteredInferenceProvider(inner=provider, collector=collector)
    return LlmQueryRewriter(metered, ModelId("qwen3:1.7b"), logger)


async def test_rewrite_sends_dialogue_and_question_to_model(
    logger: structlog.stdlib.BoundLogger,
    tmp_path: Path,
) -> None:
    provider = MockInferenceProvider(response_content="перенос отпуска на следующий год")
    rewriter = make_rewriter(logger, tmp_path, provider)

    rewritten = await rewriter.rewrite(REQUEST_ID, "перенести их на следующий год", TURNS)

    assert rewritten == "перенос отпуска на следующий год"
    request = provider.requests[0]
    # Тот же чат-модель, вызов без инструментов, с id запуска для метрик.
    assert request.request_id == REQUEST_ID
    assert request.model == ModelId("qwen3:1.7b")
    assert request.tools == ()
    assert [message.role.value for message in request.messages] == ["system", "user"]
    user_prompt = request.messages[1].content
    assert "Пользователь: Сколько дней отпуска?" in user_prompt
    assert "Ассистент: 28 календарных дней." in user_prompt
    assert "перенести их на следующий год" in user_prompt


async def test_empty_history_omits_dialogue_block(
    logger: structlog.stdlib.BoundLogger,
    tmp_path: Path,
) -> None:
    provider = MockInferenceProvider(response_content="перенос отпуска")
    rewriter = make_rewriter(logger, tmp_path, provider)

    rewritten = await rewriter.rewrite(REQUEST_ID, "перенести их на следующий год", ())

    assert rewritten == "перенос отпуска"
    assert "Реплики диалога" not in provider.requests[0].messages[1].content


async def test_quotes_and_whitespace_are_stripped_from_answer(
    logger: structlog.stdlib.BoundLogger,
    tmp_path: Path,
) -> None:
    provider = MockInferenceProvider(response_content="  «перенос отпуска»\n")
    rewriter = make_rewriter(logger, tmp_path, provider)

    assert await rewriter.rewrite(REQUEST_ID, "перенести их", ()) == "перенос отпуска"


async def test_empty_model_answer_falls_back_to_raw_question(
    logger: structlog.stdlib.BoundLogger,
    tmp_path: Path,
) -> None:
    provider = MockInferenceProvider(response_content="   ")
    rewriter = make_rewriter(logger, tmp_path, provider)

    assert await rewriter.rewrite(REQUEST_ID, "сырой вопрос", ()) == "сырой вопрос"


async def test_rewrite_call_is_metered_without_content(
    tmp_path: Path,
    logger: structlog.stdlib.BoundLogger,
    capturing_logger: CapturingLogger,
) -> None:
    """Дополнительный LLM-вызов попадает в метрики без содержимого запроса."""
    provider = MockInferenceProvider(
        response_content="перенос отпуска на следующий год",
        usage=InferenceUsage(prompt_tokens=12, completion_tokens=4),
    )
    rewriter = make_rewriter(logger, tmp_path, provider)

    await rewriter.rewrite(REQUEST_ID, "перенести их на следующий год", TURNS)

    lines = [
        json.loads(line)
        for line in (tmp_path / "metrics" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [line["kind"] for line in lines] == ["llm_call"]
    call = lines[0]
    assert call["request_id"] == "rewrite-test"
    assert call["model"] == "qwen3:1.7b"
    assert call["prompt_tokens"] == 12
    assert call["completion_tokens"] == 4
    # Ни метрики, ни structlog-дубликат не содержат текста запроса и реплик.
    assert "перенести" not in json.dumps(lines, ensure_ascii=False)
    assert "отпуск" not in json.dumps(lines, ensure_ascii=False)
    for entry in capturing_logger.calls:
        assert "перенести" not in json.dumps(entry.kwargs, ensure_ascii=False, default=str)
        assert "отпуск" not in json.dumps(entry.kwargs, ensure_ascii=False, default=str)
