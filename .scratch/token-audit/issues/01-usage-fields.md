# 01: Usage-поля в контракте инференса

**What to build:** провайдер инференса перестаёт выбрасывать счётчики токенов: ответ адаптера Ollama несёт необязательные usage-поля (токены промпта, выходные токены, длительности), спарсенные из ответа `/api/chat`. Все существующие потребители работают как раньше — поля добавляются, ничего не переезжает.

**Blocked by:** None (can start immediately).

**Status:** done

- [x] Адаптер парсит usage из консервированного ответа Ollama в доменную модель (юнит-тест)
- [x] Поля необязательны: ответ без usage обрабатывается как прежде
- [x] Все четыре гейта зелёные (`ty`, `ruff check`, `ruff format --check`, `pytest`)

## Comments

- Реализовано агентом в ветке `token-audit`, коммит `998ff3e` (feat: parse optional usage fields from Ollama chat response). Новый `InferenceUsage` (frozen dataclass) в `src/bot/inference/models.py`, парсинг в `src/bot/inference/ollama.py`; юнит-тесты в `tests/unit/inference/test_ollama.py`. Ревью Standards/Spec: жёстких нарушений нет, единственный P2 (дублирование списка полей в `_usage_to_domain`) устранён до коммита. Гейты: ty / ruff check / ruff format --check / pytest (165 passed) — зелёные.
