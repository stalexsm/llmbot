"""Судья качества свободных ответов (LLM-as-a-Judge).

Отдельный вызов модели-судьи: вопрос пользователя и ответ бота подаются
в промпт, требующий JSON-вердикт по трём критериям — вежливость, точность,
лаконичность, шкала 0..1. Парсинг вердикта под контрактом: битый вердикт —
``ValueError`` с фрагментом сырого ответа, понятная ошибка теста, а не
traceback-охота. Механика (промпт, контракт, вызов провайдера) офлайн-
тестируема; живой тест только подаёт вопросы и применяет порог.
"""

import json
from dataclasses import dataclass
from typing import Final

from bot.domain.ids import ModelId, RequestId
from bot.domain.messages import InferenceMessage, MessageRole
from bot.inference.models import InferenceRequest
from bot.inference.provider import InferenceProvider

# Критерии judge-секции задания — они же ключи JSON-вердикта.
CRITERIA: Final = ("politeness", "accuracy", "conciseness")

# Порог качества из задания: среднее по всем оценкам ≥ 0.8.
PASS_AVERAGE: Final = 0.8

# Нижняя граница таймаута вызова судьи, секунд. Вердикт — длинная генерация:
# думающая модель выдаёт сотни токенов размышлений до JSON, и на дефолтных
# OLLAMA_TIMEOUT_SECONDS=120 живой вызов пробивает таймаут вскоре после
# пяти ответов агента. Меньше настроенного таймаута пол не опускается.
MIN_TIMEOUT_SECONDS: Final = 300.0

# Сколько символов сырого ответа судьи попадает в сообщение об ошибке.
_ERROR_SNIPPET_CHARS = 200

_SYSTEM_PROMPT = (
    "Ты — строгий судья качества ответов ИИ-ассистента. Оцени ответ ассистента "
    "на вопрос пользователя по трём критериям, каждый — число от 0.0 до 1.0:\n"
    '- "politeness" — вежливость: тон уважительный и доброжелательный, нет грубости '
    "и панибратства;\n"
    '- "accuracy" — точность: ответ корректен по существу, без выдуманных фактов '
    "и уклонения от вопроса;\n"
    '- "conciseness" — лаконичность: ответ короткий и по делу, без воды и повторов, '
    "но полнота не пострадала.\n"
    "Верни только JSON-объект вида "
    '{"politeness": <число>, "accuracy": <число>, "conciseness": <число>} '
    "— без пояснений и без markdown-разметки."
)


@dataclass(frozen=True)
class Verdict:
    """Вердикт судьи: три оценки по шкале 0..1."""

    politeness: float
    accuracy: float
    conciseness: float

    @property
    def average(self) -> float:
        """Средняя оценка вердикта по трём критериям."""
        return (self.politeness + self.accuracy + self.conciseness) / len(CRITERIA)


def build_judge_messages(question: str, answer: str) -> tuple[InferenceMessage, InferenceMessage]:
    """Сообщения для вызова судьи: правила — в system, вопрос и ответ — в user."""
    user_content = f"Вопрос пользователя: {question}\nОтвет ассистента: {answer}"
    return (
        InferenceMessage(role=MessageRole.SYSTEM, content=_SYSTEM_PROMPT),
        InferenceMessage(role=MessageRole.USER, content=user_content),
    )


def parse_verdict(content: str) -> Verdict:
    """Разобрать вердикт судьи по контракту: JSON-объект с оценками 0..1.

    Markdown-оградка вокруг JSON допускается (модели часто оборачивают код,
    даже когда попросено этого не делать). Нарушение контракта — ``ValueError``
    с причиной и фрагментом сырого ответа: живой тест падает с понятной
    ошибкой, а не traceback-ом.
    """
    raw = content.strip()
    if raw.startswith("```"):
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError(_broken_verdict(content, "ожидается JSON-объект")) from None
    if not isinstance(payload, dict):
        raise ValueError(
            _broken_verdict(content, f"ожидается JSON-объект, пришёл {type(payload).__name__}")
        )
    scores: dict[str, float] = {}
    for criterion in CRITERIA:
        scores[criterion] = _score(payload, criterion, content)
    return Verdict(**scores)


async def judge_answer(
    provider: InferenceProvider,
    model: ModelId,
    question: str,
    answer: str,
    request_id: RequestId,
) -> Verdict:
    """Спросить судью о паре «вопрос — ответ» и вернуть разобранный вердикт."""
    response = await provider.generate(
        InferenceRequest(
            request_id=request_id,
            model=model,
            messages=build_judge_messages(question, answer),
        )
    )
    return parse_verdict(response.content)


def judge_timeout_seconds(configured: float) -> float:
    """Таймаут вызова судьи: настроенный, но не ниже пола для длинного вердикта."""
    return max(configured, MIN_TIMEOUT_SECONDS)


def _score(payload: dict[str, object], criterion: str, raw: str) -> float:
    """Оценка одного критерия: число в диапазоне 0..1, иначе ValueError."""
    if criterion not in payload:
        raise ValueError(_broken_verdict(raw, f"нет критерия {criterion!r}"))
    value = payload[criterion]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(_broken_verdict(raw, f"критерий {criterion!r} — не число"))
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(_broken_verdict(raw, f"критерий {criterion!r} вне шкалы 0..1: {value}"))
    return float(value)


def _broken_verdict(raw: str, reason: str) -> str:
    snippet = raw.strip()[:_ERROR_SNIPPET_CHARS]
    return f"судья вернул невалидный вердикт: {reason}; сырой ответ: {snippet!r}"
