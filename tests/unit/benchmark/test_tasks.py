"""Юнит-тесты фиксированного набора задач бенчмарка."""

from bot.benchmark.tasks import EXPECTED_KIND_COUNTS, benchmark_tasks, task_kind_counts


def test_set_has_twenty_tasks_in_stable_order() -> None:
    tasks = benchmark_tasks()
    assert len(tasks) == 20
    assert [task.task_id for task in tasks] == [task.task_id for task in benchmark_tasks()]
    assert len({task.task_id for task in tasks}) == 20


def test_kind_distribution_matches_spec() -> None:
    assert task_kind_counts(benchmark_tasks()) == EXPECTED_KIND_COUNTS


def test_every_task_has_prompt_and_checker() -> None:
    for task in benchmark_tasks():
        assert task.prompt.strip()
        assert callable(task.checker)
