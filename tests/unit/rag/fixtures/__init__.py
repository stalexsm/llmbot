"""Бинарные фикстуры PDF/DOCX для тестов извлечения (тикет 06).

``sample.pdf`` — три страницы с различимым текстом; ``sample.docx`` —
заголовки «Heading 1/2» и абзацы. Файлы коммитятся: тесты не зависят
от библиотек-генераторов.
"""

from pathlib import Path


def load(name: str) -> bytes:
    """Содержимое бинарной фикстуры по имени файла."""
    return (Path(__file__).parent / name).read_bytes()
