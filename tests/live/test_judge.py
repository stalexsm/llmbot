"""Живая оценка качества свободных ответов: судья (LLM-as-a-Judge).

Пять открытых вопросов judge-секции датасета подаются агенту headless,
ответы бота оценивает модель-судья (OLLAMA_JUDGE_MODEL, по умолчанию —
основная модель) прямым вызовом провайдера. Ассерт — среднее по всем
вердиктам ≥ PASS_AVERAGE (константа из задания); битый вердикт — понятная
ошибка теста (контракт парсинга офлайн-тестируется отдельно). Модели судьи
нет на сервере — тест скипается. Без работающего Ollama скипается вся
фикстура ``live_ollama``; маркер ``live`` ставится конфтестом автоматически.
"""

import os
from collections.abc import Callable

import pytest
import structlog.stdlib

from bot.domain.ids import ModelId, RequestId
from tests.live.dataset import DATASET_PATH, load_behavioral_dataset
from tests.live.harness import HeadlessChat
from tests.live.judge import PASS_AVERAGE, Verdict, judge_answer, judge_timeout_seconds
from tests.live.support import LiveOllama, inference_provider

_DATASET = load_behavioral_dataset(DATASET_PATH)


def _resolve_model(name: str, models: tuple[str, ...]) -> str | None:
    """Полное имя модели на сервере: точное совпадение или имя без тега версии.

    Ollama резолвит имя без тега в ``…:latest`` — для скипа и вызова судьи
    возвращается полное имя с сервера, чтобы проверка наличия и вызов не
    разошлись.
    """
    for model in models:
        if model == name or model.startswith(f"{name}:"):
            return model
    return None


async def test_judge_answers_quality(
    headless_chat_factory: Callable[[str], HeadlessChat],
    live_ollama: LiveOllama,
    logger: structlog.stdlib.BoundLogger,
) -> None:
    resolved = _resolve_model(
        os.environ.get("OLLAMA_JUDGE_MODEL") or live_ollama.model, live_ollama.models
    )
    if resolved is None:
        base = os.environ.get("OLLAMA_JUDGE_MODEL") or live_ollama.model
        pytest.skip(f"модели судьи {base} нет на сервере {live_ollama.base_url}: тест пропущен")
    # Таймаут судьи — с полом для длинного вердикта: дефолтные 120 секунд
    # думающая модель пробивает на втором-третьем вызове.
    provider = inference_provider(
        live_ollama,
        logger,
        timeout_seconds=judge_timeout_seconds(live_ollama.timeout_seconds),
    )
    chat = headless_chat_factory("judge")

    verdicts: list[Verdict] = []
    for index, question in enumerate(_DATASET.judge_questions, start=1):
        answer = await chat.send(question)
        try:
            verdict = await judge_answer(
                provider=provider,
                model=ModelId(resolved),
                question=question,
                answer=answer,
                request_id=RequestId(f"judge-q{index}"),
            )
        except ValueError as exc:
            pytest.fail(str(exc))
        verdicts.append(verdict)

    average = sum(verdict.average for verdict in verdicts) / len(verdicts)
    breakdown = "\n".join(
        f"  вопрос {index}: politeness={verdict.politeness:.2f} "
        f"accuracy={verdict.accuracy:.2f} conciseness={verdict.conciseness:.2f}"
        for index, verdict in enumerate(verdicts, start=1)
    )
    assert average >= PASS_AVERAGE, (
        f"средняя оценка судьи {average:.3f} < {PASS_AVERAGE};\n{breakdown}"
    )
