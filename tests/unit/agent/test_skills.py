"""Тесты индекса скиллов: парсинг файла, сборка из каталога, рендер в промпт."""

from pathlib import Path

import pytest
import structlog.stdlib
from structlog.testing import CapturingLogger

from bot.agent.prompts import SYSTEM_PROMPT, build_date_block, build_system_prompt
from bot.agent.skills import SkillEntry, load_skills, parse_skill, render_skills_index

_ENTRIES = (
    SkillEntry(
        name="weather",
        description="Используй для вопросов о погоде в городе.",
        file=Path("skills/weather/SKILL.md"),
    ),
    SkillEntry(
        name="notes",
        description="Заметки.",
        file=Path("skills/notes/SKILL.md"),
    ),
)


def write_skill(root: Path, name: str, text: str) -> Path:
    """Создать скилл по канонической раскладке: <root>/<имя>/SKILL.md."""
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    file = skill_dir / "SKILL.md"
    file.write_text(text, encoding="utf-8")
    return file


class TestParseSkill:
    """Табличные случаи парсинга SKILL.md.

    Шапка — YAML frontmatter с полями ``name`` и ``description``; по имени
    модель ищет файл в каталоге скиллов, по описанию понимает, когда он нужен.
    Если frontmatter нет, работает запасной вариант — строка ``Description:``.
    """

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param(
                "---\n"
                "name: weather\n"
                "Description: Используй для вопросов о погоде в городе.\n"
                "---\n"
                "## When to Use\nСпроси погоду.\n",
                id="frontmatter-полный",
            ),
            pytest.param(
                "\n---\nname: weather\ndescription: Погодная рутина: curl и сводка.\n---\n",
                id="frontmatter-нижний-регистр-ключа-значения-неважно",
            ),
        ],
    )
    def test_description_from_frontmatter(self, text: str) -> None:
        entry = parse_skill(Path("skills/weather/SKILL.md"), text)

        assert entry is not None
        assert entry.name == "weather"
        assert "погод" in entry.description.lower()

    def test_name_and_description_come_from_frontmatter(self) -> None:
        text = "---\nname: weather\ndescription: Курс валют через открытый API.\n---\n"

        entry = parse_skill(Path("skills/weather/SKILL.md"), text)

        assert entry is not None
        # Имя из шапки обязано совпадать с каталогом: по нему модель ищет файл.
        assert entry.name == "weather"
        assert entry.description == "Курс валют через открытый API."

    def test_name_mismatching_folder_is_not_indexed(self) -> None:
        """Имя из шапки расходится с каталогом — модель нашла бы не тот файл."""
        text = "---\nname: currency\ndescription: Курсы валют.\n---\n"

        assert parse_skill(Path("skills/weather/SKILL.md"), text) is None

    def test_fallback_to_plain_description_line(self) -> None:
        entry = parse_skill(
            Path("skills/weather/SKILL.md"),
            "Description: Запасной формат без frontmatter.\n\nШаги…\n",
        )

        assert entry is not None
        assert entry.name == "weather"  # нет имени в шапке — имя каталога
        assert entry.description == "Запасной формат без frontmatter."

    @pytest.mark.parametrize(
        ("text", "expected_description"),
        [
            pytest.param(
                "Description: Используй для вопросов о погоде в городе.\n\n"
                "## When to Use\nСпроси погоду.\n",
                "Используй для вопросов о погоде в городе.",
                id="описание-затем-секции",
            ),
            pytest.param(
                "\n\nDescription: Погодная рутина: curl и сводка.\n\n## Commands\ncurl …\n",
                "Погодная рутина: curl и сводка.",
                id="пустые-строки-до-описания",
            ),
            pytest.param(
                "description: поле в нижнем регистре тоже описание.\n",
                "поле в нижнем регистре тоже описание.",
                id="префикс-в-нижнем-регистре",
            ),
            pytest.param(
                "# Заголовок сверху не мешает\n\nDescription: Прогноз для города.\n\nШаги…\n",
                "Прогноз для города.",
                id="заголовок-до-описания",
            ),
            pytest.param(
                "Description:   Описание с пробелами по краям.  ",
                "Описание с пробелами по краям.",
                id="пробелы-по-краям-обрезаются",
            ),
        ],
    )
    def test_description_is_the_value_of_the_field(
        self, text: str, expected_description: str
    ) -> None:
        entry = parse_skill(Path("skills/weather/SKILL.md"), text)

        assert entry is not None
        assert entry.name == "weather"
        assert entry.description == expected_description

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("", id="пустой-файл"),
            pytest.param("# Только заголовок, полей нет\n", id="без-описания"),
            pytest.param(
                "---\nname: weather\n---\nТело без описания.\n", id="frontmatter-без-description"
            ),
            pytest.param("Description:\n", id="поле-без-значения"),
            pytest.param("Мысли о погоде, но самого поля нет.\n", id="обычный-текст"),
        ],
    )
    def test_no_description_means_no_skill(self, text: str) -> None:
        assert parse_skill(Path("skills/weather/SKILL.md"), text) is None

    def test_name_and_file_come_from_path(self) -> None:
        entry = parse_skill(Path("skills/weather/SKILL.md"), "Description: Погода в Минске.\n")

        assert entry is not None
        assert entry.name == "weather"
        assert entry.file == Path("skills/weather/SKILL.md")


class TestLoadSkills:
    """Сборка индекса из каталога: раскладка, порядок, фильтры, устойчивость."""

    def test_entries_sorted_by_name(
        self, tmp_path: Path, logger: structlog.stdlib.BoundLogger
    ) -> None:
        write_skill(tmp_path, "zz-last", "Description: Последний.\n")
        write_skill(tmp_path, "aa-first", "Description: Первый.\n")

        entries = load_skills(tmp_path, logger)

        assert [entry.name for entry in entries] == ["aa-first", "zz-last"]
        assert entries[0].file == tmp_path / "aa-first" / "SKILL.md"

    def test_stray_files_and_folders_without_skill_md_are_ignored(
        self, tmp_path: Path, logger: structlog.stdlib.BoundLogger
    ) -> None:
        write_skill(tmp_path, "real", "Description: Настоящий скилл.\n")
        # Мусор в корне каталога скиллов.
        (tmp_path / "notes.txt").write_text("не скилл\n", encoding="utf-8")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        (tmp_path / "readme.md").write_text("тоже не скилл\n", encoding="utf-8")
        # Каталог без SKILL.md — не скилл.
        empty = tmp_path / "empty-skill"
        empty.mkdir()
        (empty / "TODO.txt").write_text("нет файла SKILL.md\n", encoding="utf-8")

        assert [entry.name for entry in load_skills(tmp_path, logger)] == ["real"]

    def test_skill_without_description_is_skipped(
        self, tmp_path: Path, logger: structlog.stdlib.BoundLogger
    ) -> None:
        write_skill(tmp_path, "good", "Description: Хороший скилл.\n")
        write_skill(tmp_path, "broken", "# Только заголовок\n")

        assert [entry.name for entry in load_skills(tmp_path, logger)] == ["good"]

    def test_unreadable_or_not_utf8_skill_is_skipped_with_warning(
        self,
        tmp_path: Path,
        logger: structlog.stdlib.BoundLogger,
        capturing_logger: CapturingLogger,
    ) -> None:
        """Битый файл не роняет старт: скилл пропускается, в лог уходит warning с путём."""
        write_skill(tmp_path, "good", "Description: Хороший скилл.\n")
        bad = tmp_path / "bad" / "SKILL.md"
        bad.parent.mkdir()
        bad.write_bytes(b"\xff\xfe\xff\x00 garbage")

        entries = load_skills(tmp_path, logger)

        assert [entry.name for entry in entries] == ["good"]
        failures = [
            call
            for call in capturing_logger.calls
            if call.kwargs.get("event") == "skill_load_failed"
        ]
        assert len(failures) == 1
        assert failures[0].method_name == "warning"
        assert failures[0].kwargs["file"] == str(bad)

    def test_missing_directory_gives_empty_index(
        self, tmp_path: Path, logger: structlog.stdlib.BoundLogger
    ) -> None:
        assert load_skills(tmp_path / "нет-такого", logger) == ()

    def test_new_folder_appears_in_index_on_next_load(
        self, tmp_path: Path, logger: structlog.stdlib.BoundLogger
    ) -> None:
        """Новый скилл подхватывается без правки кода: повторная сборка его видит."""
        write_skill(tmp_path, "weather", "Description: Погода в Минске.\n")
        assert [entry.name for entry in load_skills(tmp_path, logger)] == ["weather"]

        write_skill(tmp_path, "notes", "Description: Заметки пользователя.\n")

        assert [entry.name for entry in load_skills(tmp_path, logger)] == ["notes", "weather"]

    def test_repo_weather_skill_parses(
        self, repo_root: Path, logger: structlog.stdlib.BoundLogger
    ) -> None:
        """Закоммиченный скилл погоды лежит по канону и читается."""
        entries = load_skills(repo_root / "skills", logger)

        weather = next(entry for entry in entries if entry.name == "weather")
        assert weather.file == repo_root / "skills" / "weather" / "SKILL.md"
        assert "погод" in weather.description.lower()


class TestRenderSkillsIndex:
    """Рендер секции индекса для системного промпта."""

    def test_index_lists_names_and_descriptions(self) -> None:
        section = render_skills_index(_ENTRIES)

        assert "- weather: Используй для вопросов о погоде в городе." in section
        assert "- notes: Заметки." in section

    def test_footer_tells_where_skill_files_live(self) -> None:
        section = render_skills_index(_ENTRIES)

        # Путь в примере выводится из имени реальной записи, а не зашит литералом.
        assert "каталоге скиллов" in section
        assert "cat skills/weather/SKILL.md" in section

    def test_index_tells_model_to_read_file_via_exec(self) -> None:
        section = render_skills_index(_ENTRIES)

        assert "exec" in section
        assert "cat" in section

    def test_empty_index_renders_nothing(self) -> None:
        assert render_skills_index(()) == ""

    def test_only_first_sentence_of_description_in_index(self) -> None:
        entry = (
            SkillEntry(
                name="weather",
                description=(
                    "Используй для вопросов о погоде в городе. Живые данные не отвечай по памяти."
                ),
                file=Path("skills/weather/SKILL.md"),
            ),
        )

        section = render_skills_index(entry)

        assert "Используй для вопросов о погоде в городе." in section
        assert "не отвечай по памяти" not in section

    def test_long_single_sentence_is_shortened(self) -> None:
        entry = (
            SkillEntry(
                name="long",
                description="Очень-описательное " * 30 + "предложение без конца",
                file=Path("skills/long/SKILL.md"),
            ),
        )

        section = render_skills_index(entry)

        listed = next(line for line in section.splitlines() if line.startswith("- long:"))
        assert len(listed) < 250
        assert listed.endswith("…")


class TestBuildSystemPrompt:
    """Композиция системного промпта: база + секция скиллов."""

    def test_without_skills_equals_base_prompt(self) -> None:
        assert build_system_prompt("") == SYSTEM_PROMPT

    def test_with_skills_appends_section(self) -> None:
        section = render_skills_index(_ENTRIES)

        prompt = build_system_prompt(section)

        assert prompt.startswith(SYSTEM_PROMPT)
        assert section in prompt

    def test_base_prompt_is_self_sufficient_without_skills(self) -> None:
        """Без скиллов промпт не ссылается на них."""
        assert "скилл" not in SYSTEM_PROMPT.lower()


class TestPromptBudget:
    """Бюджет системного промпта (оптимизация 3: сжатие по данным аудита).

    Промпт уходит в каждый шаг каждого запуска, поэтому его размер — прямая
    статья расхода токенов. Гард держит статичную часть (база, блок даты)
    в пределах бюджета символов; qwen3 токенизирует русский примерно в
    2.4 символа на токен. Индекс скиллов не лимитируется: он растёт
    с числом скиллов и управляется владельцем, а не кодом.
    """

    _BASE_BUDGET_CHARS = 1150
    _DATE_BUDGET_CHARS = 150

    def test_base_prompt_fits_budget(self) -> None:
        assert len(SYSTEM_PROMPT) <= self._BASE_BUDGET_CHARS, (
            f"системный промпт раздулся: {len(SYSTEM_PROMPT)} > {self._BASE_BUDGET_CHARS} символов"
        )

    def test_date_block_fits_budget(self) -> None:
        from datetime import datetime

        block = build_date_block(datetime(2026, 8, 27, 14, 30))

        assert len(block) <= self._DATE_BUDGET_CHARS

    def test_semantic_anchors_survive_compression(
        self, repo_root: Path, logger: structlog.stdlib.BoundLogger
    ) -> None:
        """Сжатие не должно выкидывать поведенчески критичные правила."""
        skills = load_skills(repo_root / "skills", logger)
        prompt = build_system_prompt(render_skills_index(skills))

        # Точная фраза честного отказа: её ищет ApplicationService.
        assert "Я не знаю точного ответа на этот вопрос" in prompt
        # Имя инструмента и анти-эхо правило.
        assert "execute_command" in prompt
        assert "stdout" in prompt
        # Поиск по документам: имя инструмента, Источник и честное «не нашёл».
        assert "search_documents" in prompt
        assert "Источник" in prompt
        assert "не нашёл" in prompt
        # Чтение скилла файлом через exec с примером пути.
        assert "cat skills/weather/SKILL.md" in prompt
        # Живые данные — только через инструмент.
        assert "Живые данные" in prompt


class TestBuildDateBlock:
    """Блок актуальной даты: рендерится на каждый запуск цикла."""

    def test_fixed_moment_renders_full_date_and_weekday(self) -> None:
        from datetime import datetime

        moment = datetime(2026, 8, 27, 14, 30)  # четверг

        block = build_date_block(moment)

        assert block == (
            "<date>Сегодня: 27 августа 2026, четверг. "
            "Точные дату и время — командой execute_command (date).</date>"
        )

    def test_block_is_asked_fresh_per_call(self) -> None:
        from datetime import datetime

        morning = build_date_block(datetime(2026, 8, 27, 9, 0))
        evening = build_date_block(datetime(2026, 8, 28, 21, 0))

        assert "27 августа" in morning
        assert "28 августа" in evening
