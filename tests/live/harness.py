"""Headless-чат для поведенческих тестов: агентный цикл с чат-сессией в БД.

По образцу headless-раннера бенчмарка: кейс подаётся агенту без Telegram.
Отличие от бенчмарка — диалог живёт в чат-сессии во временной БД, как у
настоящего чата: история предыдущих реплик подмешивается к каждому запросу,
а сброс сессии работает как команда /new. Механика диалога — офлайн-
тестируемый шов: она не зависит от живой модели.
"""

from pathlib import Path
from typing import Protocol

import structlog

from bot.agent.exec import ExecTool
from bot.agent.loop import AgentLoop, AgentRun
from bot.agent.progress import AgentProgress
from bot.agent.prompts import build_system_prompt
from bot.agent.skills import load_skills, render_skills_index
from bot.config.settings import Settings
from bot.domain.ids import ModelId, RequestId, TelegramChatId
from bot.domain.messages import InferenceMessage, MessageRole
from bot.domain.tools import ExecutionContext
from bot.inference.provider import InferenceProvider
from bot.sessions.store import ChatSessionStore
from bot.sessions.window import trim_to_window


class AgentRunner(Protocol):
    """Шов запуска агентного цикла: всё, что нужно диалогу headless-чата."""

    async def run(
        self,
        request_id: RequestId,
        history: tuple[InferenceMessage, ...],
        user_message: InferenceMessage,
        progress: AgentProgress | None = None,
        context: ExecutionContext | None = None,
    ) -> AgentRun: ...


class HeadlessChat:
    """Диалог с агентным циклом через чат-сессию — как в application-слое.

    ``send`` повторяет шов обработки сообщения без Telegram: окно истории из
    хранилища, запуск цикла, в сессию — только реплики диалога (tool-обмен
    хранилище отфильтровывает само). Запуск без финального ответа — ошибка:
    сессия остаётся на последнем удачном ходе, в датасете нет повторов.
    """

    def __init__(
        self,
        agent: AgentRunner,
        store: ChatSessionStore,
        chat_id: TelegramChatId,
        label: str,
        history_limit: int,
    ) -> None:
        self._agent = agent
        self._store = store
        self._chat_id = chat_id
        self._label = label
        self._history_limit = history_limit
        self._turn = 0

    async def send(self, text: str) -> str:
        """Отправить реплику в диалог; вернуть финальный ответ агента.

        RequestId детерминирован (``<метка>-turn<N>``), а не uuid4, как в
        пайплайне бота: по нему живой прогон трассируется в логи до кейса.
        """
        self._turn += 1
        request_id = RequestId(f"{self._label}-turn{self._turn}")
        history = trim_to_window(self._store.load(self._chat_id), self._history_limit)
        run = await self._agent.run(
            request_id,
            history,
            InferenceMessage(role=MessageRole.USER, content=text),
        )
        if run.final_answer is None or run.stopped_by_limit:
            raise RuntimeError(
                f"{self._label}: агент не дал финальный ответ на реплике {self._turn} "
                f"(шагов: {run.steps_used}, остановка по лимиту: {run.stopped_by_limit})"
            )
        self._store.append(self._chat_id, *run.exchange)
        return run.final_answer

    def reset(self) -> None:
        """Сбросить чат-сессию — эквивалент команды /new."""
        self._store.reset(self._chat_id)


def build_agent_loop(
    inference: InferenceProvider,
    model: str,
    skills_directory: Path,
    repo_root: Path,
    settings: Settings,
    logger: structlog.stdlib.BoundLogger,
) -> AgentLoop:
    """Собрать граф агента headless — как в headless-раннере бенчмарка, без метрик."""
    skills = load_skills(skills_directory, logger)
    exec_tool = ExecTool(
        cwd=repo_root,
        timeout_seconds=settings.agent_exec_timeout_seconds,
        max_output_chars=settings.agent_exec_max_output_chars,
        logger=logger,
    )
    return AgentLoop(
        inference=inference,
        model=ModelId(model),
        system_prompt=build_system_prompt(render_skills_index(skills)),
        tools=(exec_tool,),
        step_limit=settings.agent_max_steps,
        keep_steps=settings.agent_compaction_keep_steps,
        logger=logger,
    )
