"""Независимый валидатор MarkdownV2 для тестов исходящих сообщений.

Это внешний контракт разметки, по которому Telegram принимает или отвергает
сообщение; реализация экранировщика его не использует. Валидатор проверяет:
вне код-блоков и код-спанов нет неэкранированных спецсимволов, а сами
код-блоки и код-спаны сбалансированы (незакрытая конструкция отвергает
сообщение целиком).
"""

# Полный набор спецсимволов MarkdownV2 вне код-блоков и код-спанов
# (https://core.telegram.org/bots/api#markdownv2-style).
SPECIAL_CHARS = "_*[]()~`>#+-=|{}.!\\"


def assert_valid_markdown_v2(text: str) -> None:
    """Упасть, если ``text`` Telegram не принял бы как валидный MarkdownV2."""
    i, n = 0, len(text)
    in_block = False
    while i < n:
        at_line_start = i == 0 or text[i - 1] == "\n"
        if at_line_start and text.startswith("```", i):
            in_block = not in_block
            i += 3
            continue
        if in_block:
            i += 1
            continue
        char = text[i]
        if char == "\\":
            i += 2  # экранирующая пара: слеш + экранируемый символ
            continue
        if char == "`":
            end = text.find("`", i + 1)
            newline = text.find("\n", i + 1)
            balanced = end != -1 and (newline == -1 or end < newline)
            assert balanced, f"несбалансированный код-спан: {text!r}"
            i = end + 1
            continue
        assert char not in SPECIAL_CHARS, f"неэкранированный {char!r} в {text!r}"
        i += 1
    assert not in_block, f"незакрытый код-блок: {text!r}"
