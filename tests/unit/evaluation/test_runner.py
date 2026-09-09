"""Runner evaluation: попадание/промах по датасету и калибровка порога.

Поиск в runner детерминированный: эмбеддинги подменяются фейковкой
``KeywordEmbeddingProvider`` (одно ключевое слово — одна ось), поэтому
попадания и промахи известны заранее. Живой прогон против Ollama —
``python -m bot.evaluation``, он пропускается без сервера.
"""

from pathlib import Path

import pytest
import structlog
import structlog.stdlib

from bot.domain.ids import RequestId, TelegramUserId
from bot.evaluation.dataset import evaluation_documents, evaluation_questions
from bot.evaluation.runner import ChunkReport, QuestionResult, run_evaluation, suggest_threshold
from bot.rag.service import RagService
from tests.fakes import KeywordEmbeddingProvider, make_rag_service

_OWNER = TelegramUserId(7)
_KEYWORDS = ("отпуск", "командировка")


async def _indexed_service(tmp_path: Path, logger: structlog.stdlib.BoundLogger) -> RagService:
    """RagService с уже проиндексированным корпусом датасета (фейковые эмбеддинги)."""
    service = make_rag_service(tmp_path, logger, KeywordEmbeddingProvider(_KEYWORDS))
    for document in evaluation_documents():
        await service.index_document(
            RequestId(f"eval-doc-{document.name}"),
            _OWNER,
            document.name,
            document.content.encode("utf-8"),
        )
    return service


async def test_run_evaluation_reports_hit_and_miss(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    service = make_rag_service(tmp_path, logger, KeywordEmbeddingProvider(_KEYWORDS))
    report = await run_evaluation(service, _OWNER, logger)
    assert [result.question for result in report.results] == [
        item.question for item in evaluation_questions()
    ]
    for result in report.results:
        for chunk in result.chunks:
            assert isinstance(chunk, ChunkReport)
            assert chunk.snippet
    # Вопрос «мимо корпуса» обязан остаться без попаданий в чужие документы.
    negatives = [result for result in report.results if result.expected_source is None]
    assert negatives and all(not result.hit for result in negatives)


async def test_run_evaluation_hit_when_expected_document_found(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    service = make_rag_service(tmp_path, logger, KeywordEmbeddingProvider(_KEYWORDS))
    report = await run_evaluation(service, _OWNER, logger)
    positives = [result for result in report.results if result.expected_source is not None]
    assert positives, "датасет обязан содержать вопросы с известным источником"
    assert any(result.hit for result in positives)


async def test_run_evaluation_reindex_keeps_results_stable(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    """Повторный прогон заменяет документы атомарно и не меняет вердикты."""
    service = await _indexed_service(tmp_path, logger)
    first = await run_evaluation(service, _OWNER, logger)
    second = await run_evaluation(service, _OWNER, logger)
    assert [r.hit for r in first.results] == [r.hit for r in second.results]


def test_suggest_threshold_midpoint_between_hits_and_misses() -> None:
    results = (
        QuestionResult("q1", "a.md", True, (ChunkReport("a.md", 0, 0.8, "..."),)),
        QuestionResult("q2", "b.md", True, (ChunkReport("b.md", 0, 0.6, "..."),)),
        QuestionResult("q3", None, False, (ChunkReport("a.md", 1, 0.3, "..."),)),
    )
    assert suggest_threshold(results) == pytest.approx(0.45)


def test_suggest_threshold_none_when_not_separable() -> None:
    results = (
        QuestionResult("q1", "a.md", True, (ChunkReport("a.md", 0, 0.4, "..."),)),
        QuestionResult("q2", None, False, (ChunkReport("a.md", 1, 0.6, "..."),)),
    )
    assert suggest_threshold(results) is None


def test_suggest_threshold_none_without_negative_questions() -> None:
    results = (QuestionResult("q1", "a.md", True, (ChunkReport("a.md", 0, 0.7, "..."),)),)
    assert suggest_threshold(results) is None
