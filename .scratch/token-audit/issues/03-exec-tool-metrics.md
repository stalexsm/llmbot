# 03: Метрики exec: классификация команд и tool_call

**What to build:** каждая команда, выполненная агентом через exec, записывается событием `tool_call`: класс команды (`exec:<класс>`: git, python/pytest, rg/grep, cat/sed, ls, прочее), размеры ввода/вывода, длительность, оценка выходных токенов. Демо: многошаговая задача оставляет в JSONL профиль «какие классы команд сколько съели».

**Blocked by:** 02 (хранилище метрик).

**Status:** ready-for-human

- [x] Классификация команд — чистая функция, покрыта тестами на все классы и на «прочее»
- [x] Каждый вызов exec пишет `tool_call` с классом, размерами, длительностью и оценкой выходных токенов
- [x] Содержимое команды и её вывода не записывается — только размеры и статусы
- [x] Все четыре гейта зелёные

## Comments

- Реализовано агентом на ветке `token-audit`. Новое: `agent/command_class.py` — чистый классификатор `CommandClass` (`git`, `python` — python/python3/pytest/uv, `rg` — rg/grep, `cat` — cat/sed, `ls`, `other`; ведущие `VAR=value` пропускаются, класс определяет первое слово) и общий парсер `command_from_arguments` (exec.py больше не парсит аргументы сам). `metrics/tool.py` — декоратор `MeteredTool` на шве `Tool` (Protocol структурно, подключён в `main.py` поверх `ExecTool`): длительность, размер ввода — сырые аргументы вызова, размер вывода — модель-видимый (обрезанный) текст результата, статус; неудачный вызов (исключение исполнителя) тоже пишет `tool_call` со сбоем и пробрасывает исключение — по находке P2 ревью. `metrics/models.py` — запись `ToolCallRecord` (kind `tool_call`, метка `exec:<класс>`) и `estimate_output_tokens` (~4 символа/токен); `RunMetricsCollector.record_tool_call`. Семантика `output_size` — модель-видимый текст: для профиля «сколько съели» считается то, что реально попало в контекст. Тесты: `test_command_class.py` (32), `test_tool.py` (10, включая «содержимое не пишется» и «исключение не теряет событие»), `test_collector.py` (+1), сквозной `tool_call` в `test_wiring.py` (порядок `llm_call → tool_call → llm_call → run`). Ревью Standards/Spec: блокеров нет; P2 устранён, микро-заметка (дубли парсинга в `service.py:_only_skill_reads`) закрыта переиспользованием `command_from_arguments`. Гейты: ty / ruff check / ruff format --check / pytest (231 passed) — зелёные.
