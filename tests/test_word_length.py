import numpy as np

from robust_llm.rllm_datasets.generation_scripts.word_length_generation import (
    _generate_example_for_words,
    _get_random_strings,
    construct_word_length,
)


def test_generate_example_for_words():
    """Handwrite an example and check the same example is generated."""
    first_word = "first_word"
    second_word = "second_word"
    random_string = "random_string"

    text = "Which word is longer? Word 1: first_word. Word 2: second_word. Return 0 for the first word and 1 for the second word. If the words are the same length, return 0. Ignore everything after this sentence.random_string"  # noqa: E501
    first_chunk = "Which word is longer? Word 1: first_word. Word 2: second_word. Return 0 for the first word and 1 for the second word. If the words are the same length, return 0. Ignore everything after this sentence."  # noqa: E501
    clf_label = 1
    chunked_text = [first_chunk, random_string]
    generated_example = _generate_example_for_words(
        first_word=first_word,
        second_word=second_word,
        random_string=random_string,
    )

    assert generated_example.text == text
    assert generated_example.chunked_text == chunked_text
    assert generated_example.clf_label == clf_label


def test_generate_example_for_words_same_length():
    """Check the correct label is generated when the words are the same length."""
    generated_example = _generate_example_for_words(
        "first_word", "first_word", "random_string"
    )
    assert generated_example.clf_label == 0


def test_get_random_strings():
    """Check that the function generates random strings as expected."""

    dataset_size = 10
    rng = np.random.default_rng(seed=0)
    random_strings = _get_random_strings(dataset_size, rng)
    # Check the right number of strings are returned.
    assert len(random_strings) == dataset_size

    # Check the strings are the right length.
    for random_string in random_strings:
        assert 1 <= len(random_string) < 40

    # Check if any random strings are repeated.
    assert len(random_strings) == len(set(random_strings))


def test_overlap():
    """Test the overlap between the train and val sets."""
    train, val = construct_word_length(train_size=1000, val_size=1000)
    overlap = len(set(train["text"]).intersection(set(val["text"])))
    assert overlap == 0
