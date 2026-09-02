"""Юнит-тесты фиксированного набора задач бенчмарка."""

from bot.benchmark.tasks import TaskKind, benchmark_tasks, task_kind_counts

_EXPECTED_COUNTS = {
    TaskKind.KNOWLEDGE: 5,
    TaskKind.EXEC: 5,
    TaskKind.REPO: 6,
    TaskKind.MIXED: 4,
}


def test_set_has_twenty_tasks_in_stable_order() -> None:
    tasks = benchmark_tasks()
    assert len(tasks) == 20
    assert [task.task_id for task in tasks] == [task.task_id for task in benchmark_tasks()]
    assert len({task.task_id for task in tasks}) == 20


def test_kind_distribution_matches_spec() -> None:
    assert task_kind_counts(benchmark_tasks()) == _EXPECTED_COUNTS


def test_every_task_has_prompt_and_checker() -> None:
    for task in benchmark_tasks():
        assert task.prompt.strip()
        assert callable(task.checker)
