"""Загрузка поведенческого датасета: состав 13 кейсов и строгая форма записи."""

import json
from pathlib import Path

import pytest

from tests.live.dataset import (
    EXPECTED_KIND_COUNTS,
    CaseKind,
    load_behavioral_dataset,
)

_DATASET = load_behavioral_dataset()
_CASE_IDS = [case.id for case in _DATASET.cases]


def _write_cases(tmp_path: Path, cases: list[dict[str, object]]) -> Path:
    file = tmp_path / "dataset.json"
    file.write_text(json.dumps({"behavioral": cases, "judge": ["вопрос"] * 5}), encoding="utf-8")
    return file


class TestRepositoryDataset:
    def test_thirteen_cases_with_fixed_composition(self) -> None:
        assert len(_DATASET.cases) == 13
        counts = dict.fromkeys(CaseKind, 0)
        for case in _DATASET.cases:
            counts[case.kind] += 1
        assert counts == EXPECTED_KIND_COUNTS

    def test_case_ids_unique(self) -> None:
        assert len(_CASE_IDS) == len(set(_CASE_IDS))

    def test_judge_section_has_five_questions(self) -> None:
        assert len(_DATASET.judge_questions) == 5
        assert all(question.strip() for question in _DATASET.judge_questions)

    def test_single_turn_cases_have_plain_prompts(self) -> None:
        for case in _DATASET.cases:
            if case.kind in (CaseKind.JAILBREAK, CaseKind.REFUSAL):
                assert len(case.turns) == 1
                assert case.turns[0].strip()
                assert not case.entities

    def test_memory_cases_carry_entities_for_second_turn(self) -> None:
        for case in _DATASET.cases:
            if case.kind is CaseKind.MEMORY:
                assert len(case.turns) == 2
                assert len(case.entities) >= 2

    def test_reset_case_checks_entities_vanish_after_new(self) -> None:
        reset = [case for case in _DATASET.cases if case.kind is CaseKind.RESET]
        assert len(reset) == 1
        assert len(reset[0].turns) == 2
        assert reset[0].entities

    def test_homework_jailbreak_case_is_present(self) -> None:
        prompts = " ".join(
            case.turns[0] for case in _DATASET.cases if case.kind is CaseKind.JAILBREAK
        ).lower()
        assert "забуд" in prompts
        assert "системный промпт" in prompts


class TestValidation:
    def test_unknown_kind_is_value_error(self, tmp_path: Path) -> None:
        file = _write_cases(tmp_path, [{"id": "x1", "kind": "attack", "turns": ["привет"]}])
        with pytest.raises(ValueError, match="x1"):
            load_behavioral_dataset(file)

    def test_wrong_turn_count_is_value_error(self, tmp_path: Path) -> None:
        file = _write_cases(
            tmp_path,
            [{"id": "m1", "kind": "memory", "turns": ["шаг один"], "entities": ["раз", "два"]}],
        )
        with pytest.raises(ValueError, match="m1"):
            load_behavioral_dataset(file)

    def test_memory_without_entities_is_value_error(self, tmp_path: Path) -> None:
        file = _write_cases(
            tmp_path,
            [{"id": "m2", "kind": "memory", "turns": ["шаг один", "шаг два"]}],
        )
        with pytest.raises(ValueError, match="m2"):
            load_behavioral_dataset(file)

    def test_empty_turn_is_value_error(self, tmp_path: Path) -> None:
        file = _write_cases(
            tmp_path,
            [{"id": "j1", "kind": "jailbreak", "turns": ["   "]}],
        )
        with pytest.raises(ValueError, match="j1"):
            load_behavioral_dataset(file)

    def test_wrong_case_total_is_value_error(self, tmp_path: Path) -> None:
        cases: list[dict[str, object]] = [
            {"id": f"j{i}", "kind": "jailbreak", "turns": ["привет"]} for i in range(5)
        ]
        cases.append({"id": "extra", "kind": "refusal", "turns": ["лишний"]})
        for i in range(4):
            cases.append({"id": f"r{i}", "kind": "refusal", "turns": ["вопрос"]})
        for i in range(3):
            cases.append(
                {"id": f"m{i}", "kind": "memory", "turns": ["а", "б"], "entities": ["в", "г"]}
            )
        cases.append({"id": "s0", "kind": "reset", "turns": ["а", "б"], "entities": ["в"]})
        file = _write_cases(tmp_path, cases)
        with pytest.raises(ValueError, match="13"):
            load_behavioral_dataset(file)
