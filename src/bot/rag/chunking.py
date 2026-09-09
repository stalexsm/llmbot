"""Структурный чанкинг извлечённого текста.

Границы — абзацы и заголовки Markdown (строка с ``#``): заголовок начинает
новый блок и остаётся внутри своего чанка как контекст. Мелкие блоки
склеиваются до целевого размера, слишком длинный блок режется по границам
слов/строк; между соседними чанками остаётся overlap — хвост предыдущего
чанка, срезанный по границе слова. Чистые функции без I/O: параметры приходят
из конфигурации, покрытие — прямыми unit-тестами (как тесты окна истории).
"""

import re

# ATX-заголовок Markdown: от одного до шести «#» и пробел.
_HEADING_RE = re.compile(r"^#{1,6}\s")

# Разделитель блоков внутри чанка: двойной перевод строки, как в абзацах.
_BLOCK_SEPARATOR = "\n\n"
# Overlap пристёгивается переводом строки: разрыв пришёлся на середину текста.
_OVERLAP_SEPARATOR = "\n"


def chunk_text(text: str, *, target_chars: int = 900, overlap_chars: int = 150) -> list[str]:
    """Разбить текст на чанки-строки.

    ``target_chars`` — целевой размер чанка (верхняя граница для склейки
    блоков), ``overlap_chars`` — запас перекрытия соседних чанков. Чанк не
    длиннее ``target_chars + overlap_chars + 1``: цель приблизительная, границы
    — по блокам и словам, а не по счёту символов.
    """
    if target_chars < 1:
        raise ValueError("target_chars must be positive")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must not be negative")
    if overlap_chars >= target_chars:
        raise ValueError("overlap_chars must be less than target_chars")

    blocks = _split_blocks(text)
    pieces = _merge_blocks(_force_heading_boundaries(blocks), target_chars)
    return _apply_overlap(pieces, overlap_chars)


def _split_blocks(text: str) -> list[str]:
    """Нарезать текст на блоки: абзацы и фрагменты от заголовка до абзаца."""
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            if current:
                blocks.append(_BLOCK_SEPARATOR.join(current).strip())
                current = []
            continue
        if _HEADING_RE.match(line) and current:
            blocks.append(_BLOCK_SEPARATOR.join(current).strip())
            current = [line.rstrip()]
            continue
        current.append(line.rstrip())
    if current:
        blocks.append(_BLOCK_SEPARATOR.join(current).strip())
    return [block for block in blocks if block]


def _force_heading_boundaries(blocks: list[str]) -> list[str]:
    """Заголовок не остаётся в хвосте предыдущего куска: он начинает свой.

    Иначе заголовок описывает чужой чанк и портит эмбеддинг обоих.
    """
    pieces: list[str] = []
    current: list[str] = []
    for block in blocks:
        if _HEADING_RE.match(block) and current:
            pieces.append(_BLOCK_SEPARATOR.join(current))
            current = []
        current.append(block)
    if current:
        pieces.append(_BLOCK_SEPARATOR.join(current))
    return pieces


def _merge_blocks(blocks: list[str], target_chars: int) -> list[str]:
    """Склеить блоки в куски не длиннее целевого размера; длинные — отрезать."""
    pieces: list[str] = []
    current: list[str] = []
    length = 0
    for block in blocks:
        if len(block) > target_chars:
            if current:
                pieces.append(_BLOCK_SEPARATOR.join(current))
                current, length = [], 0
            pieces.extend(_split_long_block(block, target_chars))
            continue
        extra = len(block) + (len(_BLOCK_SEPARATOR) if current else 0)
        if current and length + extra > target_chars:
            pieces.append(_BLOCK_SEPARATOR.join(current))
            current, length = [], 0
            extra = len(block)
        current.append(block)
        length += extra
    if current:
        pieces.append(_BLOCK_SEPARATOR.join(current))
    return pieces


def _split_long_block(block: str, target_chars: int) -> list[str]:
    """Разрезать один блок длиннее целевого размера по границам слов и строк."""
    pieces: list[str] = []
    start = 0
    while start < len(block):
        end = min(start + target_chars, len(block))
        if end == len(block):
            pieces.append(block[start:end].strip())
            break
        cut = _find_break(block, start, end)
        if cut is None:
            pieces.append(block[start:end].strip())
            start = end
            continue
        pieces.append(block[start:cut].strip())
        start = cut + 1  # разделитель остаётся в предыдущем куске
    return [piece for piece in pieces if piece]


def _find_break(text: str, start: int, end: int) -> int | None:
    """Последняя граница слов/строк в окне; ``None`` — если границы нет."""
    window = text[start:end]
    newline = window.rfind("\n")
    if newline != -1:
        return start + newline
    space = window.rfind(" ")
    if space != -1:
        return start + space
    return None


def _apply_overlap(pieces: list[str], overlap_chars: int) -> list[str]:
    """Пристегнуть к каждому куску хвост предыдущего (срез по границе слова)."""
    if overlap_chars == 0 or len(pieces) <= 1:
        return pieces
    chunks = [pieces[0]]
    for piece in pieces[1:]:
        tail = _overlap_tail(chunks[-1], overlap_chars)
        if tail:
            chunks.append(tail + _OVERLAP_SEPARATOR + piece)
        else:
            chunks.append(piece)
    return chunks


def _overlap_tail(previous: str, overlap_chars: int) -> str:
    """Хвост предыдущего чанка: до ~``overlap_chars`` символов, с целого слова."""
    if len(previous) <= overlap_chars:
        return previous
    candidate = previous[-overlap_chars:]
    for offset, char in enumerate(candidate):
        if char in (" ", "\n"):
            return candidate[offset + 1 :]
    return candidate
