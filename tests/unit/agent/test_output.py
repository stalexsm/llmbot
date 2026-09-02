"""Unit tests for cleaning command output: ANSI codes and progress noise."""

from bot.agent.output import clean_output, truncate_middle


def test_clean_text_passes_through_unchanged() -> None:
    text = "exit_code: 0\nstdout:\nпривет\tмир\n"
    assert clean_output(text) == text


def test_strips_ansi_colors_and_styles() -> None:
    noisy = "\x1b[32mOK\x1b[0m \x1b[1;31mFAIL\x1b[0m"
    assert clean_output(noisy) == "OK FAIL"


def test_strips_cursor_movement_and_erase_sequences() -> None:
    noisy = "\x1b[2K\x1b[1A\x1b[2Kскачивание завершено"
    assert clean_output(noisy) == "скачивание завершено"


def test_strips_osc_title_sequences() -> None:
    assert clean_output("\x1b]0;моё окно\x07готово") == "готово"


def test_collapses_carriage_return_progress_frames() -> None:
    noisy = "10%\r50%\r100%\nготово\n"
    assert clean_output(noisy) == "100%\nготово\n"


def test_keeps_final_frame_when_progress_line_ends_with_carriage_return() -> None:
    assert clean_output("50%\r100%\r\nдалее") == "100%\nдалее"


def test_strips_spinner_characters() -> None:
    assert clean_output("⠋ установка\n⠹ установка") == " установка\n установка"


def test_applies_backspaces_inside_a_frame() -> None:
    assert clean_output("100%\b\b\b\bготово") == "готово"


def test_removes_stray_control_characters() -> None:
    assert clean_output("a\x07b\x00c") == "abc"


def test_removes_stray_escape_and_c1_controls() -> None:
    # Обрывочный ESC без аргументов и C1-диапазон — тоже шум.
    assert clean_output("итог\x1b") == "итог"
    assert clean_output("a\x85b\x9bc") == "abc"


LONG_OUTPUT = (
    "начало вывода\n" + "a" * 300 + "\n" + "b" * 2000 + "\n" + "c" * 300 + "\nконец вывода"
)


def test_short_text_passes_through_unchanged() -> None:
    assert truncate_middle("короткий вывод", 200) == "короткий вывод"


def test_long_text_keeps_head_and_tail_within_limit() -> None:
    result = truncate_middle(LONG_OUTPUT, 1000)

    assert len(result) <= 1000
    assert result.startswith("начало вывода\n")
    assert result.endswith("конец вывода")
    assert "aaa" in result
    assert "ccc" in result
    assert "b" * 500 not in result


def test_truncation_marker_names_omitted_size() -> None:
    result = truncate_middle(LONG_OUTPUT, 1000)

    assert "пропущено" in result
    assert "символов" in result


def test_respects_small_limit() -> None:
    # Минимум настройки лимита: AGENT_EXEC_MAX_OUTPUT_CHARS имеет ge=200.
    result = truncate_middle(LONG_OUTPUT, 200)

    assert 0 < len(result) <= 200
    assert result.startswith("начало вывода\n")
    assert result.endswith("конец вывода")


def test_tiny_limit_still_respected() -> None:
    # Защитный случай: лимит меньше маркера — допустимо лишь не превысить лимит.
    assert 0 < len(truncate_middle("x" * 500, 10)) <= 10
