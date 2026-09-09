"""CLI живого прогона evaluation: ``uv run python -m bot.evaluation``.

Прогон идёт против живого Ollama с bge-m3: корпус датасета индексируется
в одноразовую rag-БД во временном каталоге, по каждому вопросу печатаются
найденные чанки с близостью и вердикт «попадание/промах», в конце — сводка
и предложенный порог «не найдено». Без запущенного Ollama или без модели
прогон честно пропускается: печатается причина и код возврата 0.
"""

import argparse
import asyncio
import contextlib
import logging
import sys
import tempfile
from pathlib import Path

import httpx
import structlog
from pydantic import SecretStr

from bot.config.settings import Settings
from bot.domain.ids import ModelId, TelegramUserId
from bot.evaluation.runner import EvaluationReport, run_evaluation
from bot.inference.embeddings_ollama import OllamaEmbeddingProvider
from bot.rag.migrations import apply_migrations
from bot.rag.service import RagService
from bot.rag.store import RagStore


class EvaluationSettings(Settings):
    """Settings живого прогона evaluation: Telegram-токен не нужен."""

    telegram_bot_token: SecretStr = SecretStr("")


def configure_logging() -> structlog.stdlib.BoundLogger:
    """Тихий лог: structlog-события прогона уходят в stderr, отчёт — в stdout."""
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    return structlog.stdlib.get_logger()


async def _check_live(base_url: str, embed_model: str) -> str | None:
    """Причина пропуска, если Ollama недоступен или без модели эмбеддингов; иначе ``None``."""
    try:
        async with httpx.AsyncClient(timeout=2.0, trust_env=False) as client:
            response = await client.get(f"{base_url}/api/tags")
            models = [model.get("name", "") for model in response.json().get("models", [])]
    except (httpx.HTTPError, ValueError):
        return f"Ollama недоступен по адресу {base_url}"
    if not any(name.startswith(embed_model) for name in models):
        return f"модель {embed_model} не скачана: выполните `ollama pull {embed_model}`"
    return None


async def run_evaluation_live(settings: Settings, logger: structlog.stdlib.BoundLogger) -> int:
    """Живой прогон датасета; код возврата 0 (включая честный пропуск)."""
    skip_reason = await _check_live(settings.ollama_base_url, settings.ollama_embed_model)
    if skip_reason is not None:
        sys.stdout.write(f"=== Evaluation пропущен: {skip_reason} ===\n")
        return 0
    http_client = httpx.AsyncClient(trust_env=False)
    try:
        with tempfile.TemporaryDirectory(prefix="rag-evaluation-") as directory:
            database = Path(directory) / "rag.db"
            apply_migrations(database)
            service = RagService(
                store=RagStore(database=database, logger=logger),
                embeddings=OllamaEmbeddingProvider(
                    client=http_client,
                    base_url=settings.ollama_base_url,
                    timeout_seconds=settings.ollama_embed_timeout_seconds,
                    logger=logger,
                ),
                model=ModelId(settings.ollama_embed_model),
                chunk_target_chars=settings.rag_chunk_target_chars,
                chunk_overlap_chars=settings.rag_chunk_overlap_chars,
                top_k=settings.rag_search_top_k,
                overfetch=settings.rag_search_overfetch,
                min_similarity=settings.rag_min_similarity,
                max_file_bytes=settings.rag_max_file_bytes,
                max_text_chars=settings.rag_max_text_chars,
                max_chunks=settings.rag_max_chunks,
                logger=logger,
            )
            report = await run_evaluation(service, TelegramUserId(0), logger)
    finally:
        await http_client.aclose()
    _print_report(report)
    return 0


def _print_report(report: EvaluationReport) -> None:
    lines: list[str] = ["=== Evaluation: по вопросам ==="]
    for result in report.results:
        verdict = "ПОПАДАНИЕ" if result.hit else "ПРОМАХ"
        expected = result.expected_source or "— (ответа в корпусе нет)"
        lines.append(f"[{verdict}] {result.question}")
        lines.append(f"  ожидаемый источник: {expected}")
        if result.chunks:
            for chunk in result.chunks:
                page = f", стр. {chunk.page}" if chunk.page is not None else ""
                lines.append(
                    f"  · {chunk.document}#{chunk.position}{page}"
                    f" (близость {chunk.similarity:.3f}): {chunk.snippet}…"
                )
        else:
            lines.append("  · ничего не найдено (ниже порога близости)")
    lines.append("=== Сводка ===")
    lines.append(
        f"Вопросов: {len(report.results)}, попаданий: {report.hit_count}"
        f" ({report.hit_count / len(report.results):.0%})"
    )
    if report.suggested_threshold is not None:
        lines.append(
            f"Предложенный порог «не найдено» (середина разделяющей полосы): "
            f"{report.suggested_threshold}"
        )
    else:
        lines.append("Порог не переоценивается: результаты не разделяются полосой близости.")
    sys.stdout.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Прогон RAG evaluation против живого Ollama")
    parser.parse_args()
    logger = configure_logging()
    settings = EvaluationSettings()  # pydantic-settings: значения RAG_* из .env/окружения
    exit_code = asyncio.run(run_evaluation_live(settings, logger))
    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
