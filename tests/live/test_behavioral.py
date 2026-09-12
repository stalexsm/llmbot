"""Поведенческие тесты: 13 кейсов датасета против живого агента.

Кейс подаётся агенту headless через агентный цикл (по образцу раннера
бенчмарка), чат-сессия кейса живёт во временной БД; ответ сверяется
предикатом-чекером из ``checkers.py``. Один прогон, строгий ассерт:
повторов нет, флакующий кейс правится в датасете, а не ослаблением
ассерта. Без работающего Ollama кейсы скипаются (фикстура ``live_ollama``);
маркер ``live`` ставится конфтестом автоматически.
"""

import os
from collections.abc import Callable

import pytest

from tests.live.checkers import honest_refusal, jailbreak_clean, mentions_all, mentions_none
from tests.live.dataset import DATASET_PATH, BehavioralCase, CaseKind, load_behavioral_dataset
from tests.live.harness import HeadlessChat

_DATASET = load_behavioral_dataset(DATASET_PATH)
_CASES = _DATASET.cases


def _secret_markers(case: BehavioralCase) -> tuple[str, ...]:
    """Реальные секреты окружения как дополнительные запрещённые маркеры.

    Токен бота загружен фикстурой ``live_ollama`` из .env (тест получает её
    через ``headless_chat_factory``): утечка именно этого значения — самый
    точный признак провала кейса.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    return (*case.forbid, token) if token else case.forbid


@pytest.mark.parametrize("case", _CASES, ids=[case.id for case in _CASES])
async def test_behavioral_case(
    case: BehavioralCase,
    headless_chat_factory: Callable[[str], HeadlessChat],
) -> None:
    chat = headless_chat_factory(case.id)
    match case.kind:
        case CaseKind.JAILBREAK:
            answer = await chat.send(case.turns[0])
            assert jailbreak_clean(answer, forbid=_secret_markers(case)), answer
        case CaseKind.REFUSAL:
            answer = await chat.send(case.turns[0])
            assert honest_refusal(answer, accept=case.accept), answer
        case CaseKind.MEMORY:
            await chat.send(case.turns[0])
            answer = await chat.send(case.turns[1])
            assert mentions_all(answer, case.entities), answer
        case CaseKind.RESET:
            await chat.send(case.turns[0])
            chat.reset()
            answer = await chat.send(case.turns[1])
            assert mentions_none(answer, case.entities), answer
