"""Unit-тесты структурного чанкинга: границы, склейка, overlap (Шов 4).

Вторая секция — ``chunk_pages``: чанкинг постранично с привязкой чанка
PDF к странице (тикет 06).
"""

import pytest

from bot.rag.chunking import chunk_pages, chunk_text
from bot.rag.models import PageText


def make_paragraphs(count: int, size: int, *, mark: str = "абзац") -> str:
    """Текст из ``count`` абзацев примерно по ``size`` символов каждый.

    Слова пронумерованы: повторяющийся текст даёт ложные совпадения
    при проверке перекрытия соседних чанков.
    """
    vocabulary = [f"слово{number}" for number in range((size // 9) + 1)]
    paragraphs = []
    for number in range(count):
        paragraphs.append(f"{mark} {number}: " + " ".join(vocabulary))
    return "\n\n".join(paragraphs)


def common_overlap_length(previous: str, current: str) -> int:
    """Длина общего суффикса/префикса соседних чанков (0 — нет перекрытия)."""
    limit = min(len(previous), len(current))
    for length in range(limit, 0, -1):
        if previous[-length:] == current[:length]:
            return length
    return 0


def test_short_text_is_single_chunk() -> None:
    text = "Короткий текст без разбивки на абзацы."

    assert chunk_text(text, target_chars=900, overlap_chars=150) == [text]


def test_paragraphs_merge_up_to_target_size() -> None:
    # 20 абзацев по ~120 символов: склеиваются до цели ~900;
    # с overlap чанк не длиннее цели плюс запас перекрытия.
    text = make_paragraphs(20, 120)
    chunks = chunk_text(text, target_chars=900, overlap_chars=150)

    assert len(chunks) >= 3
    assert all(len(chunk) <= 900 + 150 + 1 for chunk in chunks)
    assert all(len(chunk) >= 200 for chunk in chunks[:-1])


def test_neighbouring_chunks_share_overlap() -> None:
    text = make_paragraphs(20, 120)
    chunks = chunk_text(text, target_chars=900, overlap_chars=150)

    assert len(chunks) > 1
    overlaps = [
        common_overlap_length(chunks[index], chunks[index + 1]) for index in range(len(chunks) - 1)
    ]
    # Хвост срезан по границе слова, поэтому чуть меньше 150, но не пустой.
    assert all(50 < overlap <= 150 for overlap in overlaps)


def test_overlap_prefix_starts_at_word_boundary() -> None:
    text = make_paragraphs(20, 120)
    chunks = chunk_text(text, target_chars=900, overlap_chars=150)

    for chunk in chunks[1:]:
        # Пристёгивание идёт через перевод строки; хвост начинается с целого
        # слова — первый символ не разделитель.
        assert chunk.split("\n", 1)[0][0] not in (" ", "\n")


def test_long_paragraph_is_hard_split_with_overlap() -> None:
    # Один абзац на несколько тысяч символов: режется по словам на чанки.
    paragraph = " ".join(f"слово{index}" for index in range(1000))
    chunks = chunk_text(paragraph, target_chars=900, overlap_chars=150)

    assert len(chunks) >= 5
    assert all(len(chunk) <= 900 + 150 + 1 for chunk in chunks)
    overlaps = [
        common_overlap_length(chunks[index], chunks[index + 1]) for index in range(len(chunks) - 1)
    ]
    assert all(50 < overlap <= 150 for overlap in overlaps)


def test_heading_starts_new_chunk_content_and_stays_inside() -> None:
    first = make_paragraphs(4, 200)
    second_body = make_paragraphs(4, 200, mark="тело")
    text = f"{first}\n\n# Отпуск\n\n{second_body}"

    chunks = chunk_text(text, target_chars=900, overlap_chars=150)

    # Заголовок начинает новое содержимое своего чанка, а не болтается
    # в хвосте чужого: с телом раздела он в одном чанке, предыдущий чанк
    # им не заканчивается.
    assert any("# Отпуск\n\nтело" in chunk for chunk in chunks)
    assert not any(chunk.rstrip().endswith("# Отпуск") for chunk in chunks)


def test_no_line_is_lost() -> None:
    lines = [f"строка {number}: " + " ".join(["текст"] * 15) for number in range(60)]
    text = "\n\n".join(lines)

    chunks = chunk_text(text, target_chars=600, overlap_chars=100)

    remaining = "\n".join(chunks)
    for line in lines:
        assert line in remaining


@pytest.mark.parametrize("text", ["", "   \n\n  \n"])
def test_empty_text_yields_no_chunks(text: str) -> None:
    assert chunk_text(text, target_chars=900, overlap_chars=150) == []


def test_custom_small_parameters() -> None:
    text = make_paragraphs(10, 40)

    chunks = chunk_text(text, target_chars=60, overlap_chars=10)

    assert len(chunks) > 1
    assert all(len(chunk) <= 60 + 10 + 1 for chunk in chunks)


def test_zero_overlap_disables_overlap() -> None:
    text = make_paragraphs(20, 120)

    chunks = chunk_text(text, target_chars=900, overlap_chars=0)

    assert len(chunks) > 1
    assert all(
        common_overlap_length(chunks[index], chunks[index + 1]) == 0
        for index in range(len(chunks) - 1)
    )


@pytest.mark.parametrize(
    ("target", "overlap"),
    [(0, 0), (100, 100), (100, 150), (-5, 0), (900, -1)],
)
def test_invalid_parameters_fail_fast(target: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("текст", target_chars=target, overlap_chars=overlap)


# --- chunk_pages: постраничный чанкинг с привязкой к странице (тикет 06) ---


def test_chunk_pages_numbers_chunks_sequentially_across_pages() -> None:
    pages = [
        PageText(page=1, text=make_paragraphs(8, 200)),
        PageText(page=2, text=make_paragraphs(8, 200, mark="вторая")),
    ]

    chunks = chunk_pages(pages, target_chars=300, overlap_chars=40)

    assert [chunk.position for chunk in chunks] == list(range(len(chunks)))
    assert len(chunks) > 2


def test_chunk_pages_attach_page_numbers_and_never_mix_pages() -> None:
    pages = [
        PageText(page=3, text=make_paragraphs(8, 200)),
        PageText(page=7, text=make_paragraphs(8, 200, mark="седьмая")),
    ]

    chunks = chunk_pages(pages, target_chars=300, overlap_chars=40)

    # Номер страницы — у страницы, а не по порядку чанков; чанк не содержит
    # текст чужой страницы, значит Источник точен до страницы.
    third = [chunk for chunk in chunks if chunk.page == 3]
    seventh = [chunk for chunk in chunks if chunk.page == 7]
    assert third and seventh
    assert all("седьмая" not in chunk.text for chunk in third)
    assert all("седьмая" in chunk.text for chunk in seventh)


def test_chunk_pages_without_page_numbers_for_plain_documents() -> None:
    chunks = chunk_pages([PageText(page=None, text=make_paragraphs(4, 120))])

    assert chunks
    assert all(chunk.page is None for chunk in chunks)


def test_chunk_pages_skips_empty_pages() -> None:
    pages = [PageText(page=1, text=""), PageText(page=2, text="Один абзац.")]

    chunks = chunk_pages(pages)

    assert [chunk.page for chunk in chunks] == [2]
    assert [chunk.text for chunk in chunks] == ["Один абзац."]
