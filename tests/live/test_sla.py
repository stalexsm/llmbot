"""Живой SLA-тест задержки: полный ответ не дольше бюджета.

Один короткий канонический вызов напрямую к провайдеру — мимо агентного
цикла, Telegram и чат-сессии (прецедент прямого вызова — живой тест судьи).
Ассерт один, из задания: полный ответ укладывается в OLLAMA_LATENCY_SECONDS
(дефолт 4). Прокси-репорт времени обработки промпта — в вывод теста; он же
попадает в сообщение ассерта, так что при падении цифры видны всегда,
без ``-s``. Без работающего Ollama фикстура ``live_ollama`` скипает тест;
маркер ``live`` ставится конфтестом автоматически.
"""

import sys

import structlog.stdlib

from bot.domain.ids import ModelId, RequestId
from bot.inference.models import InferenceRequest
from tests.live.sla import (
    build_canonical_messages,
    build_sla_report,
    latency_budget_seconds,
    measure_generate,
)
from tests.live.support import LiveOllama, inference_provider


async def test_full_answer_within_latency_budget(
    live_ollama: LiveOllama,
    logger: structlog.stdlib.BoundLogger,
) -> None:
    provider = inference_provider(live_ollama, logger)
    # Бюджет читается из окружения, куда live_ollama уже разложила .env:
    # фикстура обязана быть запрошена раньше этого вызова (сигнатура выше).
    budget_seconds = latency_budget_seconds()
    request = InferenceRequest(
        request_id=RequestId("sla-latency"),
        model=ModelId(live_ollama.model),
        messages=build_canonical_messages(),
    )

    response, elapsed_seconds = await measure_generate(provider, request)
    report = build_sla_report(
        elapsed_seconds=elapsed_seconds,
        budget_seconds=budget_seconds,
        response=response,
    )
    # Прокси-репорт — в вывод теста (T20 запрещает print; sys.stdout.write
    # виден с -s и в захваченном выводе при падении). Ассерта на прокси нет:
    # отступление от буквы задания объяснено в докстринге tests/live/sla.py —
    # инференс нестриминговый, TTFT неизмерим, прокси — время обработки
    # промпта из ответа Ollama.
    sys.stdout.write(report + "\n")

    assert elapsed_seconds <= budget_seconds, (
        f"полный ответ {elapsed_seconds:.3f} с превысил бюджет "
        f"{budget_seconds:.3f} с (OLLAMA_LATENCY_SECONDS)\n{report}"
    )
