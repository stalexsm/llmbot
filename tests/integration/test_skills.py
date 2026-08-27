"""Интеграционные сценарии скиллов: индекс в промпте → чтение файла → команда.

Стек собирается как в композиционном корне (load_skills → render →
build_system_prompt → AgentLoop с настоящим ExecTool в tmp-каталоге),
но провайдер — скриптованный: ответы модели заготовлены заранее.
Настоящая сеть не нужна.
"""

from pathlib import Path
from uuid import uuid4

import structlog.stdlib

from bot.agent.exec import ExecTool
from bot.agent.loop import AgentLoop
from bot.agent.prompts import build_system_prompt
from bot.agent.skills import load_skills, render_skills_index
from bot.domain.ids import ModelId, RequestId
from bot.domain.messages import InferenceMessage, MessageRole
from tests.fakes import (
    RecordingProgress,
    ScriptedInferenceProvider,
    exec_call_response,
    final_response,
)

_SKILL_TEXT = """---
name: weather
description: Тестовый скилл погоды для интеграционного сценария.
---

## Commands

1. Выполни команду: echo weather-json-ok
2. Ответь пользователю сводкой по-русски.
"""


def make_skills_dir(tmp_path: Path) -> Path:
    skill_dir = tmp_path / "skills" / "weather"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(_SKILL_TEXT, encoding="utf-8")
    return tmp_path / "skills"


async def test_agent_reads_skill_and_executes_its_command(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    """Сценарий «погода»: агент находит скилл в индексе, читает файл, выполняет команду."""
    skills = make_skills_dir(tmp_path)
    system_prompt = build_system_prompt(render_skills_index(load_skills(skills)))
    provider = ScriptedInferenceProvider(
        [
            exec_call_response("cat skills/weather/SKILL.md"),
            exec_call_response("echo weather-json-ok"),
            final_response("Погода в Минске: облачно, +19, ветер 14 км/ч"),
        ]
    )
    exec_tool = ExecTool(cwd=tmp_path, timeout_seconds=5.0, max_output_chars=4000, logger=logger)
    loop = AgentLoop(
        inference=provider,
        model=ModelId("qwen3:1.7b"),
        system_prompt=system_prompt,
        tools=(exec_tool,),
        step_limit=10,
        logger=logger,
    )
    progress = RecordingProgress()
    question = InferenceMessage(role=MessageRole.USER, content="Какая погода в Минске?")

    run = await loop.run(RequestId(str(uuid4())), (), question, progress)

    # Индекс в промпте: имя, описание и путь — но не полный текст скилла.
    system_message = provider.requests[0].messages[0]
    assert system_message.role is MessageRole.SYSTEM
    assert "weather" in system_message.content
    assert "skills/weather/SKILL.md" in system_message.content
    assert "Тестовый скилл погоды" in system_message.content
    assert "## Commands" not in system_message.content
    assert "echo weather-json-ok" not in system_message.content

    # Шаг 1: агент прочитал файл скилла настоящим exec — видит полную рутину.
    read_result = run.exchange[2]
    assert read_result.role is MessageRole.TOOL
    assert "## Commands" in read_result.content
    assert "echo weather-json-ok" in read_result.content

    # Шаг 2: агент выполнил команду из скилла — вывод виден модели.
    command_result = run.exchange[4]
    assert command_result.role is MessageRole.TOOL
    assert "weather-json-ok" in command_result.content
    assert "exit_code: 0" in command_result.content

    assert run.final_answer == "Погода в Минске: облачно, +19, ветер 14 км/ч"
    assert run.stopped_by_limit is False
    assert progress.events == [
        ("started", "cat skills/weather/SKILL.md", None),
        ("finished", "cat skills/weather/SKILL.md", True),
        ("started", "echo weather-json-ok", None),
        ("finished", "echo weather-json-ok", True),
    ]


async def test_new_skill_file_joins_index_without_code_changes(
    logger: structlog.stdlib.BoundLogger, tmp_path: Path
) -> None:
    """Скилл, добавленный файлом, виден в индексе при следующей сборке промпта."""
    skills = make_skills_dir(tmp_path)

    first_prompt = build_system_prompt(render_skills_index(load_skills(skills)))
    assert "currency" not in first_prompt

    currency_dir = skills / "currency"
    currency_dir.mkdir()
    (currency_dir / "SKILL.md").write_text(
        "Description: Курс валют через открытый API.\n\n## Commands\n\n"
        "Выполни curl-запрос курса.\n",
        encoding="utf-8",
    )
    second_prompt = build_system_prompt(render_skills_index(load_skills(skills)))

    assert "currency" in second_prompt
    assert "skills/currency/SKILL.md" in second_prompt
    assert "Курс валют через открытый API." in second_prompt
    # Полные рутины по-прежнему не попадают в промпт.
    assert "curl-запрос курса" not in second_prompt
