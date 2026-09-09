"""Датасет evaluation: полнота и согласованность с корпусом документов."""

from bot.evaluation.dataset import evaluation_documents, evaluation_questions


def test_dataset_has_at_least_five_questions_with_known_sources() -> None:
    questions = evaluation_questions()
    assert len(questions) >= 5
    for item in questions:
        assert item.question.strip()
        assert item.expected_source is None or item.expected_source.strip()


def test_expected_sources_reference_dataset_documents() -> None:
    names = {document.name for document in evaluation_documents()}
    for item in evaluation_questions():
        assert item.expected_source is None or item.expected_source in names


def test_dataset_documents_are_non_empty_and_unique() -> None:
    documents = evaluation_documents()
    assert len(documents) >= 2
    names = [document.name for document in documents]
    assert len(names) == len(set(names))
    for document in documents:
        assert len(document.content.strip()) > 200


def test_dataset_includes_negative_question() -> None:
    """Калибровка порога «не найдено» требует вопроса мимо корпуса."""
    assert any(item.expected_source is None for item in evaluation_questions())
