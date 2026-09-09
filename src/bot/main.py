"""Composition root: builds the runtime object graph and starts the bot.

All runtime objects are created here and injected explicitly; no module-level
runtime state exists anywhere else. Lifecycle ownership is explicit: the HTTP
client and the Telegram bot session are closed in ``finally`` blocks.
"""

import asyncio
import contextlib
import logging
import sys
from pathlib import Path

import httpx
import structlog
from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession

from bot.agent.exec import ExecTool
from bot.agent.loop import AgentLoop
from bot.agent.prompts import build_system_prompt
from bot.agent.skills import load_skills, render_skills_index
from bot.application.documents import DocumentService
from bot.application.search import SearchDocumentsTool
from bot.application.service import ApplicationService
from bot.config.settings import Settings
from bot.domain.ids import ModelId, RequestId
from bot.domain.messages import InferenceMessage, MessageRole
from bot.inference.embeddings_ollama import OllamaEmbeddingProvider
from bot.inference.models import InferenceRequest
from bot.inference.ollama import OllamaInferenceProvider
from bot.inference.provider import InferenceProvider
from bot.metrics.collector import RunMetricsCollector
from bot.metrics.provider import MeteredInferenceProvider
from bot.metrics.recorder import METRICS_DIRECTORY, MetricsRecorder
from bot.metrics.tool import MeteredTool
from bot.rag.migrations import RAG_DATABASE_PATH
from bot.rag.migrations import apply_migrations as apply_rag_migrations
from bot.rag.service import RagService
from bot.rag.store import RagStore
from bot.sessions.migrations import CHAT_DATABASE_PATH, apply_migrations
from bot.sessions.store import ChatSessionStore
from bot.telegram.handlers import BOT_COMMANDS, TelegramHandlers
from bot.telegram.loader import TelegramDocumentLoader


def configure_logging(settings: Settings) -> structlog.stdlib.BoundLogger:
    """Configure structlog once, in the composition root, and return the base logger."""
    logging.basicConfig(stream=sys.stdout, level=settings.log_level.upper(), format="%(message)s")
    renderer: structlog.typing.Processor = (
        structlog.dev.ConsoleRenderer()
        if settings.log_format == "console"
        else structlog.processors.JSONRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            renderer,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    return structlog.stdlib.get_logger()


_REWRITE_SYSTEM_PROMPT = (
    "Ты готовишь поисковые запросы по документам. Тебе даны последние реплики "
    "диалога и поисковый запрос, который может содержать местоимения и ссылки "
    "на диалог. Построй по ним самостоятельный поисковый запрос — полную "
    "формулировку без местоимений и ссылок на диалог. В ответе только текст "
    "запроса, одной строкой, без пояснений и кавычек."
)


class LlmQueryRewriter:
    """Реализация порта rag-слоя QueryRewriter поверх провайдера инференса.

    Тот же чат-модель, что и агентный цикл: один короткий вызов строит
    самостоятельный Поисковый запрос из вопроса и реплик диалога. Вызов идёт
    через декоратор метрик, поэтому попадает в общий учёт llm_call; содержимое
    запроса и реплик в метрики и лог не попадает. Пустой ответ — не сбой,
    а повод искать по сырому вопросу; исключение от провайдера уходит
    вызывающей стороне (инструмент переведёт поиск на сырой запрос).
    """

    def __init__(
        self,
        inference: InferenceProvider,
        model: ModelId,
        logger: structlog.stdlib.BoundLogger,
    ) -> None:
        self._inference = inference
        self._model = model
        self._logger = logger.bind(component="query_rewriter")

    async def rewrite(
        self,
        request_id: RequestId,
        question: str,
        recent_turns: tuple[str, ...],
    ) -> str:
        """Переписать вопрос в самостоятельный Поисковый запрос."""
        parts = []
        if recent_turns:
            parts.append("Реплики диалога:\n" + "\n".join(recent_turns))
        parts.append(f"Поисковый запрос: {question}")
        parts.append("Самостоятельный поисковый запрос:")
        response = await self._inference.generate(
            InferenceRequest(
                request_id=request_id,
                model=self._model,
                messages=(
                    InferenceMessage(role=MessageRole.SYSTEM, content=_REWRITE_SYSTEM_PROMPT),
                    InferenceMessage(role=MessageRole.USER, content="\n\n".join(parts)),
                ),
            )
        )
        rewritten = response.content.strip().strip("\"'«»„“").strip()
        if not rewritten:
            self._logger.info("query_rewrite_empty", request_id=request_id)
            return question
        return rewritten


async def run() -> None:
    settings = Settings()  # pydantic-settings validates configuration and fails fast
    logger = configure_logging(settings)

    # trust_env=False: клиент ходит в локальный Ollama напрямую. Иначе httpx
    # подхватывает прокси из переменных окружения или системных настроек macOS,
    # запросы к localhost уходят в прокси-клиент и рвутся (503 на каждый запрос).
    http_client = httpx.AsyncClient(
        timeout=settings.ollama_timeout_seconds,
        trust_env=False,
    )
    try:
        inference = OllamaInferenceProvider(
            client=http_client,
            base_url=settings.ollama_base_url,
            timeout_seconds=settings.ollama_timeout_seconds,
            logger=logger,
            think=settings.ollama_think,
            num_ctx=settings.ollama_num_ctx,
        )
        # Учёт токенов: декоратор на шве InferenceProvider пишет llm_call на
        # каждый вызов модели; сервис закрывает запуск записью run.
        metrics_collector = RunMetricsCollector(
            recorder=MetricsRecorder(directory=METRICS_DIRECTORY, logger=logger),
            input_price_per_mtok=settings.metrics_input_price_per_mtok,
            output_price_per_mtok=settings.metrics_output_price_per_mtok,
        )
        inference = MeteredInferenceProvider(
            inner=inference,
            collector=metrics_collector,
        )
        exec_tool = ExecTool(
            cwd=Path.cwd(),
            timeout_seconds=settings.agent_exec_timeout_seconds,
            max_output_chars=settings.agent_exec_max_output_chars,
            logger=logger,
        )
        # Учёт вызовов инструментов: декоратор на шве Tool пишет tool_call
        # на каждый вызов (exec — с классом команды, поиск — под своим именем).
        metered_exec_tool = MeteredTool(inner=exec_tool, collector=metrics_collector)
        # Схемы БД — только миграции alembic (ADR-0003): применяются один раз
        # на старте; сбой роняет процесс до старта polling (fail fast). БД
        # чат-сессий и rag-БД — отдельные файлы, у каждого слоя свои миграции.
        apply_migrations(CHAT_DATABASE_PATH)
        apply_rag_migrations(RAG_DATABASE_PATH)

        # Переписывание поискового запроса (conversation-aware поиск): порт
        # объявлен в rag-слое, реализация — тот же чат-модель поверх учёта
        # токенов, сборка здесь, в композиционном корне.
        query_rewriter = LlmQueryRewriter(inference, ModelId(settings.ollama_model), logger)

        # RAG: эмбеддинги — тот же httpx-клиент, отдельный таймаут /api/embed;
        # сервис документов сериализует загрузки одного владельца.
        rag_service = RagService(
            store=RagStore(database=RAG_DATABASE_PATH, logger=logger),
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
        documents = DocumentService(rag_service, logger)
        # Инструмент поиска по документам: скоуп владельца и реплики диалога
        # приходят контекстом выполнения из цикла (ADR-0002), перед поиском
        # запрос переписывается; в метрики — только размеры и статус.
        metered_search_tool = MeteredTool(
            inner=SearchDocumentsTool(query_rewriter, rag_service, logger),
            collector=metrics_collector,
        )
        # Индекс скиллов собирается один раз на старте: новый файл попадёт
        # в индекс при следующем запуске, без правки кода.
        skills = load_skills(settings.agent_skills_directory, logger)
        if not settings.agent_skills_directory.is_dir():
            logger.warning(
                "skills_directory_missing",
                directory=str(settings.agent_skills_directory),
            )
        logger.info(
            "skills_index_built",
            directory=str(settings.agent_skills_directory),
            count=len(skills),
            names=[entry.name for entry in skills],
        )
        agent_loop = AgentLoop(
            inference=inference,
            model=ModelId(settings.ollama_model),
            system_prompt=build_system_prompt(render_skills_index(skills)),
            tools=(metered_exec_tool, metered_search_tool),
            step_limit=settings.agent_max_steps,
            keep_steps=settings.agent_compaction_keep_steps,
            logger=logger,
        )

        service = ApplicationService(
            agent=agent_loop,
            sessions=ChatSessionStore(
                database=CHAT_DATABASE_PATH,
                history_limit=settings.agent_history_max_messages,
                logger=logger,
            ),
            history_limit=settings.agent_history_max_messages,
            logger=logger,
            metrics=metrics_collector,
        )

        # aiogram expects a plain numeric timeout here: during polling it
        # computes `session.timeout + polling_timeout` for long-poll requests.
        session = AiohttpSession(timeout=settings.telegram_timeout_seconds)
        bot = Bot(
            token=settings.telegram_bot_token.get_secret_value(),
            session=session,
        )
        try:
            dispatcher = Dispatcher()
            TelegramHandlers(
                service=service,
                documents=documents,
                document_loader=TelegramDocumentLoader(),
                logger=logger,
                allowed_chat_ids=settings.telegram_allowed_chat_ids,
            ).register(dispatcher)
            # Меню команд (подсветка по «/» с описаниями): сбой роняет
            # процесс до старта polling — без меню бот тоже неполноценен.
            await bot.set_my_commands(list(BOT_COMMANDS))
            logger.info(
                "bot_started",
                model=settings.ollama_model,
                allowed_chats=len(settings.telegram_allowed_chat_ids),
            )
            await dispatcher.start_polling(bot)
        finally:
            await bot.session.close()
    finally:
        await http_client.aclose()


def main() -> None:
    with contextlib.suppress(KeyboardInterrupt, SystemExit):
        asyncio.run(run())


if __name__ == "__main__":
    main()
