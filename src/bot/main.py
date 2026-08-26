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
from bot.agent.prompts import SYSTEM_PROMPT
from bot.application.service import ApplicationService
from bot.config.settings import Settings
from bot.domain.ids import ModelId
from bot.inference.ollama import OllamaInferenceProvider
from bot.sessions.store import ChatSessionStore
from bot.telegram.handlers import TelegramHandlers


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


async def run() -> None:
    settings = Settings()  # pydantic-settings validates configuration and fails fast
    logger = configure_logging(settings)

    http_client = httpx.AsyncClient(timeout=settings.ollama_timeout_seconds)
    try:
        inference = OllamaInferenceProvider(
            client=http_client,
            base_url=settings.ollama_base_url,
            timeout_seconds=settings.ollama_timeout_seconds,
            logger=logger,
        )
        exec_tool = ExecTool(
            cwd=Path.cwd(),
            timeout_seconds=settings.agent_exec_timeout_seconds,
            max_output_chars=settings.agent_exec_max_output_chars,
            logger=logger,
        )
        agent_loop = AgentLoop(
            inference=inference,
            model=ModelId(settings.ollama_model),
            system_prompt=SYSTEM_PROMPT,
            tools=(exec_tool,),
            step_limit=settings.agent_max_steps,
            logger=logger,
        )
        service = ApplicationService(
            agent=agent_loop,
            sessions=ChatSessionStore(directory=Path(".data/chats"), logger=logger),
            history_limit=settings.agent_history_max_messages,
            logger=logger,
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
            TelegramHandlers(service=service, logger=logger).register(dispatcher)
            logger.info("bot_started", model=settings.ollama_model)
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
