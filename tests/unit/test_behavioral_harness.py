"""Headless-чат: диалоговая механика поверх агентного цикла и чат-сессии."""

from pathlib import Path

import pytest
import structlog.stdlib

from bot.agent.loop import AgentLoop, AgentRun
from bot.config.settings import Settings
from bot.domain.ids import RequestId, TelegramChatId
from bot.domain.messages import InferenceMessage, MessageRole
from bot.sessions.migrations import apply_migrations
from bot.sessions.store import ChatSessionStore
from tests.fakes import MockInferenceProvider
from tests.live.harness import HeadlessChat, build_agent_loop

# Настройки с дефолтами полей: model_construct не читает окружение и не
# требует токен — юнит-тесту ручки графа нужны только как данные.
_SETTINGS = Settings.model_construct()


class ScriptedAgent:
    """Фейк агентного цикла: отвечает по заготовкам, запоминает запросы."""

    def __init__(self, answers: list[str], *, tool_exchange: bool = False) -> None:
        self._answers = list(answers)
        self._tool_exchange = tool_exchange
        self.calls: list[tuple[RequestId, tuple[InferenceMessage, ...], str]] = []

    async def run(
        self,
        request_id: RequestId,
        history: tuple[InferenceMessage, ...],
        user_message: InferenceMessage,
        progress: object = None,
        context: object = None,
    ) -> AgentRun:
        self.calls.append((request_id, history, user_message.content))
        answer = self._answers.pop(0)
        if not answer:
            return AgentRun(
                steps_used=3,
                exchange=(user_message,),
                final_answer=None,
                stopped_by_limit=True,
            )
        if self._tool_exchange:
            exchange = (
                user_message,
                InferenceMessage(role=MessageRole.ASSISTANT, content="", tool_calls=()),
                InferenceMessage(role=MessageRole.TOOL, content="stdout"),
                InferenceMessage(role=MessageRole.ASSISTANT, content=answer),
            )
        else:
            exchange = (
                user_message,
                InferenceMessage(role=MessageRole.ASSISTANT, content=answer),
            )
        return AgentRun(
            steps_used=1, exchange=exchange, final_answer=answer, stopped_by_limit=False
        )


def make_store(
    tmp_path: Path,
    logger: structlog.stdlib.BoundLogger,
) -> ChatSessionStore:
    """Реальный SQLite-store над мигрированной временной БД."""
    database = tmp_path / "chats.db"
    apply_migrations(database)
    return ChatSessionStore(database=database, logger=logger)


def make_chat(
    agent: ScriptedAgent,
    tmp_path: Path,
    logger: structlog.stdlib.BoundLogger,
    label: str = "case-x",
) -> HeadlessChat:
    return HeadlessChat(
        agent=agent,
        store=make_store(tmp_path, logger),
        chat_id=TelegramChatId(700),
        label=label,
        history_limit=20,
    )


async def test_send_returns_final_answer(
    tmp_path: Path,
    logger: structlog.stdlib.BoundLogger,
) -> None:
    agent = ScriptedAgent(["первый ответ", "второй ответ"])
    chat = make_chat(agent, tmp_path, logger)

    assert await chat.send("вопрос 1") == "первый ответ"
    assert await chat.send("вопрос 2") == "второй ответ"


async def test_second_send_receives_first_dialogue_as_history(
    tmp_path: Path,
    logger: structlog.stdlib.BoundLogger,
) -> None:
    agent = ScriptedAgent(["ответ 1", "ответ 2"])
    chat = make_chat(agent, tmp_path, logger)
    await chat.send("вопрос 1")

    await chat.send("вопрос 2")

    _, history, _ = agent.calls[1]
    assert [(message.role, message.content) for message in history] == [
        (MessageRole.USER, "вопрос 1"),
        (MessageRole.ASSISTANT, "ответ 1"),
    ]


async def test_tool_exchange_is_not_persisted(
    tmp_path: Path,
    logger: structlog.stdlib.BoundLogger,
) -> None:
    agent = ScriptedAgent(["ответ 1", "ответ 2"], tool_exchange=True)
    chat = make_chat(agent, tmp_path, logger)

    await chat.send("вопрос 1")
    await chat.send("вопрос 2")

    _, history, _ = agent.calls[1]
    assert [message.role for message in history] == [MessageRole.USER, MessageRole.ASSISTANT]


async def test_reset_clears_history(
    tmp_path: Path,
    logger: structlog.stdlib.BoundLogger,
) -> None:
    agent = ScriptedAgent(["ответ 1", "ответ 2"])
    chat = make_chat(agent, tmp_path, logger)
    await chat.send("вопрос 1")

    chat.reset()
    await chat.send("вопрос 2")

    _, history, _ = agent.calls[1]
    assert history == ()


async def test_stopped_run_raises_and_session_kept_intact(
    tmp_path: Path,
    logger: structlog.stdlib.BoundLogger,
) -> None:
    agent = ScriptedAgent(["ответ 1", "", "ответ 3"])
    chat = make_chat(agent, tmp_path, logger)
    await chat.send("вопрос 1")

    with pytest.raises(RuntimeError, match="case-x"):
        await chat.send("сломанный вопрос")

    await chat.send("вопрос 3")
    _, history, _ = agent.calls[2]
    assert [(message.role, message.content) for message in history] == [
        (MessageRole.USER, "вопрос 1"),
        (MessageRole.ASSISTANT, "ответ 1"),
    ]


async def test_request_ids_derive_from_label_and_turn(
    tmp_path: Path,
    logger: structlog.stdlib.BoundLogger,
) -> None:
    agent = ScriptedAgent(["ответ 1", "ответ 2"])
    chat = make_chat(agent, tmp_path, logger, label="jb-03")

    await chat.send("вопрос 1")
    await chat.send("вопрос 2")

    assert [str(call[0]) for call in agent.calls] == ["jb-03-turn1", "jb-03-turn2"]


async def test_build_agent_loop_builds_real_loop(
    tmp_path: Path,
    logger: structlog.stdlib.BoundLogger,
) -> None:
    agent = build_agent_loop(
        inference=MockInferenceProvider(),
        model="qwen3:4b",
        skills_directory=tmp_path / "skills",
        repo_root=tmp_path,
        settings=_SETTINGS,
        logger=logger,
    )
    assert isinstance(agent, AgentLoop)
