"""Поведенческий датасет: чтение JSON и строгая проверка формы записи.

Датасет — ``test_dataset.json`` рядом с этим модулем, две секции из спеки:
``behavioral`` (13 кейсов: 5 джейлбрейков, 4 отказа от ответа, 3 памяти,
1 сброс /new) и ``judge`` (5 открытых вопросов — их забирает судья, тикеты
уровня 3). Кейсы пополняются правкой JSON, код не меняется; форма записи
проверяется при загрузке: битый кейс — ValueError с идентификатором кейса,
а не тихое искажение прогона.
"""

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

DATASET_PATH = Path(__file__).resolve().parent / "test_dataset.json"


class CaseKind(StrEnum):
    """Вид поведенческого кейса: какая устойчивость агента проверяется."""

    JAILBREAK = "jailbreak"
    REFUSAL = "refusal"
    MEMORY = "memory"
    RESET = "reset"


# Целевой состав набора (инвариант спеки): 5 джейлбрейков, 4 отказа,
# 3 памяти, 1 сброс /new.
EXPECTED_KIND_COUNTS: dict[CaseKind, int] = {
    CaseKind.JAILBREAK: 5,
    CaseKind.REFUSAL: 4,
    CaseKind.MEMORY: 3,
    CaseKind.RESET: 1,
}

# Судья оценивает ответы на 5 открытых вопросов (константа задания).
EXPECTED_JUDGE_QUESTIONS = 5

# Число реплик по видам кейсов: джейлбрейк и отказ — одна атака/вопрос,
# память и сброс — «шаг 1 с сущностями, шаг 2 с вопросом по ним».
_TURNS_BY_KIND: dict[CaseKind, int] = {
    CaseKind.JAILBREAK: 1,
    CaseKind.REFUSAL: 1,
    CaseKind.MEMORY: 2,
    CaseKind.RESET: 2,
}


@dataclass(frozen=True)
class BehavioralCase:
    """Один кейс датасета: реплики для агента и параметры чекера.

    ``entities`` — основы слов из шага 1, которые шаг 2 обязан назвать
    (память) или, после сброса, не назвать (reset). ``forbid`` — дополнительные
    запрещённые маркеры утечки, ``accept`` — дополнительные маркеры честного
    отказа (для ложных предпосылок честность часто в поправке предпосылки).
    """

    id: str
    kind: CaseKind
    turns: tuple[str, ...]
    entities: tuple[str, ...] = ()
    accept: tuple[str, ...] = ()
    forbid: tuple[str, ...] = ()


@dataclass(frozen=True)
class BehavioralDataset:
    """Загруженный датасет: поведенческие кейсы и вопросы для судьи."""

    cases: tuple[BehavioralCase, ...]
    judge_questions: tuple[str, ...]


def load_behavioral_dataset(path: Path = DATASET_PATH) -> BehavioralDataset:
    """Прочитать датасет и проверить состав: 13 кейсов по видам, 5 вопросов."""
    data = json.loads(path.read_text(encoding="utf-8"))
    cases_raw = data["behavioral"]
    if not isinstance(cases_raw, list):
        raise ValueError("behavioral section must be a list")
    cases = tuple(_parse_case(raw) for raw in cases_raw)
    _validate_composition(cases)
    questions_raw = data["judge"]
    if not isinstance(questions_raw, list):
        raise ValueError("judge section must be a list")
    questions = tuple(str(question) for question in questions_raw)
    if len(questions) != EXPECTED_JUDGE_QUESTIONS:
        raise ValueError(
            f"judge section must have {EXPECTED_JUDGE_QUESTIONS} questions, got {len(questions)}"
        )
    return BehavioralDataset(cases=cases, judge_questions=questions)


def _parse_case(raw: dict[str, object]) -> BehavioralCase:
    case_id = str(raw.get("id", "")).strip()
    if not case_id:
        raise ValueError("behavioral case without id")
    try:
        kind = CaseKind(str(raw["kind"]))
    except KeyError:
        raise ValueError(f"{case_id}: case without kind") from None
    except ValueError:
        raise ValueError(f"{case_id}: unknown kind {raw['kind']!r}") from None
    turns_raw = raw.get("turns")
    if not isinstance(turns_raw, list):
        raise ValueError(f"{case_id}: turns must be a list")
    turns = tuple(str(turn) for turn in turns_raw)
    expected_turns = _TURNS_BY_KIND[kind]
    if len(turns) != expected_turns:
        raise ValueError(f"{case_id}: kind {kind.value} expects {expected_turns} turns")
    if not all(turn.strip() for turn in turns):
        raise ValueError(f"{case_id}: empty turn")
    entities = _string_list(raw, "entities", case_id)
    accept = _string_list(raw, "accept", case_id)
    forbid = _string_list(raw, "forbid", case_id)
    if kind in (CaseKind.MEMORY, CaseKind.RESET) and not entities:
        raise ValueError(f"{case_id}: kind {kind.value} requires entities")
    if kind in (CaseKind.JAILBREAK, CaseKind.REFUSAL) and entities:
        raise ValueError(f"{case_id}: kind {kind.value} does not take entities")
    return BehavioralCase(
        id=case_id,
        kind=kind,
        turns=turns,
        entities=entities,
        accept=accept,
        forbid=forbid,
    )


def _string_list(raw: dict[str, object], key: str, case_id: str) -> tuple[str, ...]:
    """Строковый список из кейса; не-список — ValueError с идентификатором кейса."""
    value = raw.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"{case_id}: {key} must be a list")
    return tuple(str(item) for item in value)


def _validate_composition(cases: tuple[BehavioralCase, ...]) -> None:
    counts: dict[CaseKind, int] = dict.fromkeys(CaseKind, 0)
    for case in cases:
        counts[case.kind] += 1
    expected_total = sum(EXPECTED_KIND_COUNTS.values())
    if counts != EXPECTED_KIND_COUNTS or len(cases) != expected_total:
        actual = {kind.value: count for kind, count in counts.items()}
        raise ValueError(
            f"dataset composition mismatch: expected {expected_total} cases "
            f"per spec, got {len(cases)} ({actual})"
        )
