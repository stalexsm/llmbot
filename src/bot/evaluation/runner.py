"""Прогон evaluation-датасета через RagService: попадание/промах и калибровка.

Чистая логика без сети: сервис поиска внедряется снаружи — в unit-тестах это
фейковка эмбеддингов, в живом CLI-прогоне (``python -m bot.evaluation``) —
Ollama с bge-m3. Документы индексируются под одним синтетическим владельцем;
для каждого вопроса собираются найденные чанки с близостью и вердикт:
попадание — ожидаемый источник среди найденных чанков, для негативного
вопроса — пустой результат поиска («не найдено» как правильный ответ).
``suggest_threshold`` по тем же результатам предлагает порог «не найдено»:
середину между худшим попаданием и лучшим ложным срабатыванием.
"""

from dataclasses import dataclass

import structlog

from bot.domain.ids import RequestId, TelegramUserId
from bot.evaluation.dataset import evaluation_documents, evaluation_questions
from bot.rag.models import SearchHit
from bot.rag.service import RagService


@dataclass(frozen=True)
class ChunkReport:
    """Найденный чанк в отчёте: источник, позиция, близость, начало текста."""

    document: str
    position: int
    similarity: float
    snippet: str
    page: int | None = None


@dataclass(frozen=True)
class QuestionResult:
    """Итог по одному вопросу: найденные чанки, вердикт, верхняя близость."""

    question: str
    expected_source: str | None
    hit: bool
    chunks: tuple[ChunkReport, ...]

    @property
    def top_similarity(self) -> float | None:
        """Близость лучшего чанка ответа; без чанков — ``None``."""
        return self.chunks[0].similarity if self.chunks else None


@dataclass(frozen=True)
class EvaluationReport:
    """Результат прогона: вердикты по всем вопросам и предложенный порог."""

    results: tuple[QuestionResult, ...]
    suggested_threshold: float | None

    @property
    def hit_count(self) -> int:
        return sum(1 for result in self.results if result.hit)


_SNIPPET_CHARS = 120


async def run_evaluation(
    service: RagService,
    owner_id: TelegramUserId,
    logger: structlog.stdlib.BoundLogger,
) -> EvaluationReport:
    """Проиндексировать корпус датасета и прогнать все вопросы через поиск.

    Документы заменяются атомарно, поэтому повторный прогон по тому же
    сервису даёт тот же результат. Каждый вопрос — отдельный ``RequestId``
    в логах и метриках.
    """
    for index, document in enumerate(evaluation_documents()):
        await service.index_document(
            RequestId(f"eval-doc-{index}"),
            owner_id,
            document.name,
            document.content.encode("utf-8"),
        )
    results = []
    for index, item in enumerate(evaluation_questions()):
        hits = await service.search(RequestId(f"eval-question-{index}"), owner_id, item.question)
        results.append(_question_result(item.question, item.expected_source, hits))
    report = EvaluationReport(
        results=tuple(results),
        suggested_threshold=suggest_threshold(tuple(results)),
    )
    logger.info(
        "evaluation_completed",
        questions=len(report.results),
        hits=report.hit_count,
        suggested_threshold=report.suggested_threshold,
    )
    return report


def _question_result(
    question: str,
    expected_source: str | None,
    hits: tuple[SearchHit, ...],
) -> QuestionResult:
    chunks = tuple(
        ChunkReport(
            document=hit.document_name,
            position=hit.position,
            similarity=hit.similarity,
            snippet=hit.text[:_SNIPPET_CHARS],
            page=hit.page,
        )
        for hit in hits
    )
    if expected_source is None:
        # Негативный вопрос: «не найдено» — правильное поведение поиска.
        hit = not chunks
    else:
        hit = any(chunk.document == expected_source for chunk in chunks)
    return QuestionResult(
        question=question,
        expected_source=expected_source,
        hit=hit,
        chunks=chunks,
    )


def suggest_threshold(results: tuple[QuestionResult, ...]) -> float | None:
    """Порог «не найдено» по результатам прогона: середина разделяющей полосы.

    Худшая из верхних близостей вопросов-попаданий и лучшая из верхних
    близостей ложных срабатываний (негативные вопросы с ненулевым поиском,
    промахи по ожидаемому источнику) задают границы; порог — их середина.
    Полоса не разделяется или негативов с находками нет — порог не
    предлагается, текущее значение конфига остаётся без изменений.
    """
    positive_similarities = [
        result.top_similarity for result in results if result.hit and result.top_similarity
    ]
    negative_similarities = [
        result.top_similarity for result in results if not result.hit and result.top_similarity
    ]
    if not positive_similarities or not negative_similarities:
        return None
    worst_hit = min(positive_similarities)
    best_miss = max(negative_similarities)
    if best_miss >= worst_hit:
        return None
    return round((worst_hit + best_miss) / 2, 2)
