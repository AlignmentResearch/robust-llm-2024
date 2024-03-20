from robust_llm.dataset_management.word_length.word_length_dataset_generator import (
    get_word_length_datasets,
)


def test_same_seed():
    train1, val1 = get_word_length_datasets(
        train_set_size=50, validation_set_size=10, seed=0
    )
    train2, val2 = get_word_length_datasets(
        train_set_size=50, validation_set_size=10, seed=0
    )

    assert all(
        sentence1 == sentence2
        for sentence1, sentence2 in zip(train1["text"], train2["text"])
    )
    assert all(
        sentence1 == sentence2
        for sentence1, sentence2 in zip(val1["text"], val2["text"])
    )


def test_different_seed():
    train1, val1 = get_word_length_datasets(
        train_set_size=50, validation_set_size=10, seed=1
    )
    train2, val2 = get_word_length_datasets(
        train_set_size=50, validation_set_size=10, seed=2
    )

    assert all(
        sentence1 != sentence2
        for sentence1, sentence2 in zip(train1["text"], train2["text"])
    )
    assert all(
        sentence1 != sentence2
        for sentence1, sentence2 in zip(val1["text"], val2["text"])
    )
