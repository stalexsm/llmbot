"""Тесты индекса скиллов: парсинг файла, сборка из каталога, рендер в промпт."""

from pathlib import Path

import pytest

from bot.agent.prompts import SYSTEM_PROMPT, build_system_prompt
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
        text = "---\nname: currency\ndescription: Курс валют через открытый API.\n---\n"

        entry = parse_skill(Path("skills/weather/SKILL.md"), text)

        assert entry is not None
        # Имя берём из шапки: именно по нему модель ищет файл в каталоге скиллов.
        assert entry.name == "currency"
        assert entry.description == "Курс валют через открытый API."

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

    def test_entries_sorted_by_name(self, tmp_path: Path) -> None:
        write_skill(tmp_path, "zz-last", "Description: Последний.\n")
        write_skill(tmp_path, "aa-first", "Description: Первый.\n")

        entries = load_skills(tmp_path)

        assert [entry.name for entry in entries] == ["aa-first", "zz-last"]
        assert entries[0].file == tmp_path / "aa-first" / "SKILL.md"

    def test_stray_files_and_folders_without_skill_md_are_ignored(self, tmp_path: Path) -> None:
        write_skill(tmp_path, "real", "Description: Настоящий скилл.\n")
        # Мусор в корне каталога скиллов.
        (tmp_path / "notes.txt").write_text("не скилл\n", encoding="utf-8")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        (tmp_path / "readme.md").write_text("тоже не скилл\n", encoding="utf-8")
        # Каталог без SKILL.md — не скилл.
        empty = tmp_path / "empty-skill"
        empty.mkdir()
        (empty / "TODO.txt").write_text("нет файла SKILL.md\n", encoding="utf-8")

        assert [entry.name for entry in load_skills(tmp_path)] == ["real"]

    def test_skill_without_description_is_skipped(self, tmp_path: Path) -> None:
        write_skill(tmp_path, "good", "Description: Хороший скилл.\n")
        write_skill(tmp_path, "broken", "# Только заголовок\n")

        assert [entry.name for entry in load_skills(tmp_path)] == ["good"]

    def test_missing_directory_gives_empty_index(self, tmp_path: Path) -> None:
        assert load_skills(tmp_path / "нет-такого") == ()

    def test_new_folder_appears_in_index_on_next_load(self, tmp_path: Path) -> None:
        """Новый скилл подхватывается без правки кода: повторная сборка его видит."""
        write_skill(tmp_path, "weather", "Description: Погода в Минске.\n")
        assert [entry.name for entry in load_skills(tmp_path)] == ["weather"]

        write_skill(tmp_path, "notes", "Description: Заметки пользователя.\n")

        assert [entry.name for entry in load_skills(tmp_path)] == ["notes", "weather"]

    def test_repo_weather_skill_parses(self, repo_root: Path) -> None:
        """Закоммиченный скилл погоды лежит по канону и читается."""
        entries = load_skills(repo_root / "skills")

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

        assert "skills/<имя>/SKILL.md" in section
        assert "skills/weather/SKILL.md" in section

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
