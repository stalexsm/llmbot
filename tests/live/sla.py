"""Механика SLA-теста задержки: бюджет, канонический вызов, репорт.

Инференс у бота нестриминговый — настоящий TTFT (time-to-first-token)
неизмерим: первым событием от провайдера приходит уже полный ответ.
Поэтому от буквы задания («репорт TTFT») отступаем: в вывод теста
репортится время обработки промпта (``prompt_eval_duration`` из ответа
Ollama) как прокси-TTFT — без ассерта; ассерт один, из задания: полный
ответ укладывается в OLLAMA_LATENCY_SECONDS (дефолт 4). Механика
офлайн-тестируема; живой тест только делает вызов и применяет порог.
"""

import os
import time
from collections.abc import Mapping
from typing import Final

from bot.domain.messages import InferenceMessage, MessageRole
from bot.inference.models import InferenceRequest, InferenceResponse
from bot.inference.provider import InferenceProvider

# Переменная окружения с бюджетом SLA (см. env.example, docs/runtime.md);
# живые тесты читают окружение напрямую, в обход Settings.
BUDGET_ENV_VAR: Final = "OLLAMA_LATENCY_SECONDS"

# Бюджет из задания: полный ответ не дольше 4 секунд.
DEFAULT_LATENCY_SECONDS: Final = 4.0

# Канонический короткий промпт: фиксированный и минимальный, чтобы порог
# не зависел от длины запроса. Короткость закреплена юнит-тестом.
CANONICAL_PROMPT: Final = "Ответь одним коротким словом без объяснений: столица Франции?"

_NS_PER_SECOND: Final = 1_000_000_000


def latency_budget_seconds(environ: Mapping[str, str] | None = None) -> float:
    """Бюджет SLA из OLLAMA_LATENCY_SECONDS; пусто или не задано — дефолт 4.

    Невалидное или неположительное значение — понятная ошибка, а не тихий
    дефолт: SLA, прогнанный не на том пороге, хуже упавшего теста.
    """
    env = os.environ if environ is None else environ
    raw = env.get(BUDGET_ENV_VAR, "").strip()
    if not raw:
        return DEFAULT_LATENCY_SECONDS
    try:
        budget = float(raw)
    except ValueError:
        raise ValueError(f"{BUDGET_ENV_VAR}={raw!r}: ожидалось число секунд, например 4") from None
    if budget <= 0:
        raise ValueError(f"{BUDGET_ENV_VAR}={raw!r}: бюджет должен быть положительным")
    return budget


def build_canonical_messages() -> tuple[InferenceMessage, ...]:
    """Сообщения канонического вызова: один короткий пользовательский запрос."""
    return (InferenceMessage(role=MessageRole.USER, content=CANONICAL_PROMPT),)


async def measure_generate(
    provider: InferenceProvider,
    request: InferenceRequest,
) -> tuple[InferenceResponse, float]:
    """Один вызов провайдера с замером стены: ответ и полные секунды ожидания."""
    started = time.perf_counter()
    response = await provider.generate(request)
    return response, time.perf_counter() - started


def build_sla_report(
    elapsed_seconds: float,
    budget_seconds: float,
    response: InferenceResponse,
) -> str:
    """Репорт SLA для вывода теста: полное время и разбор задержки.

    Прокси-TTFT — время обработки промпта — репортится без ассерта:
    инференс нестриминговый, настоящий TTFT неизмерим (см. докстринг
    модуля). Отсутствующие поля учёта молча пропускаются, полный отказ
    провайдера от usage репортится отдельной строкой.
    """
    lines = [
        f"SLA: полный ответ {elapsed_seconds:.3f} с из бюджета {budget_seconds:.3f} с "
        f"({BUDGET_ENV_VAR})"
    ]
    usage = response.usage
    if usage is None:
        lines.append("  провайдер не сообщил usage — прокси-TTFT не репортится")
        return "\n".join(lines)
    if usage.prompt_eval_duration_ns is not None:
        tokens = "" if usage.prompt_tokens is None else f" за {usage.prompt_tokens} ток."
        lines.append(
            f"  прокси-TTFT, обработка промпта: {_seconds(usage.prompt_eval_duration_ns):.3f} с"
            f"{tokens} — без ассерта: инференс нестриминговый, TTFT неизмерим"
        )
    if usage.load_duration_ns is not None:
        lines.append(f"  загрузка модели: {_seconds(usage.load_duration_ns):.3f} с")
    if usage.eval_duration_ns is not None:
        tokens = "" if usage.completion_tokens is None else f" за {usage.completion_tokens} ток."
        lines.append(f"  генерация ответа: {_seconds(usage.eval_duration_ns):.3f} с{tokens}")
    return "\n".join(lines)


def _seconds(nanoseconds: int) -> float:
    """Наносекунды ответа Ollama в секунды."""
    return nanoseconds / _NS_PER_SECOND
