"""Суммирование опциональных счётчиков: неизвестное (None) и ноль различаются."""

from collections.abc import Iterable


def sum_int(values: Iterable[int | None]) -> int | None:
    """Сумма счётчиков; None, если ни одного известного значения нет."""
    known = [value for value in values if value is not None]
    if not known:
        return None
    return sum(known)


def sum_float(values: Iterable[float | None]) -> float | None:
    """Сумма стоимостей; None, если ни одна не была вычислена."""
    known = [value for value in values if value is not None]
    if not known:
        return None
    return sum(known)
