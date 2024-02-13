import numpy as np

from robust_llm.configs import EnvironmentConfig
from robust_llm.dataset_management.tensor_trust.tensor_trust_dataset_generator import (
    CONTEXT_STRING,
    TWEAK_STYLES,
    WordTweaker,
    _extract_password,
    _generate_dataset,
    _modify_string,
    _shuffle_tensor_trust_dataset,
    get_tensor_trust_dataset,
    tensor_trust_get_ground_truth_label,
)
from robust_llm.pipelines.utils import _prepare_tokenizer

HAPPY_PASSWORD_STRING = CONTEXT_STRING.replace("<FIRST_TOKEN>", "myhappypassword")
TEST_STRING = "happybirthday"


def test_password_extraction():
    assert _extract_password(HAPPY_PASSWORD_STRING) == "myhappypassword"


def test_deletion():
    seed = np.random.randint(low=0, high=100)
    tweaker = WordTweaker(tweak_style="random_character_deletion", seed=seed)

    new_string = tweaker.tweak(TEST_STRING)
    assert len(new_string) == len(TEST_STRING) - 1
    for char in new_string:
        assert char in TEST_STRING

    more_strings = []
    for _ in range(10):
        new_string = tweaker.tweak(TEST_STRING)
        more_strings.append(new_string)
    more_strings_set = set(more_strings)
    assert len(more_strings_set) > 1


def test_insertion():
    seed = np.random.randint(low=0, high=100)
    tweaker = WordTweaker(tweak_style="random_character_insertion", seed=seed)
    new_string = tweaker.tweak(TEST_STRING)
    assert len(new_string) == len(TEST_STRING) + 1
    for char in TEST_STRING:
        assert char in new_string

    more_strings = []
    for _ in range(10):
        more_strings.append(tweaker.tweak(TEST_STRING))
    more_strings_set = set(more_strings)
    assert len(more_strings_set) > 5


def test_substitution():
    seed = np.random.randint(low=0, high=100)
    tweaker = WordTweaker(tweak_style="random_character_substitution", seed=seed)
    new_string = tweaker.tweak(TEST_STRING)
    assert len(new_string) == len(TEST_STRING)
    diff_count = 0
    for i, _ in enumerate(new_string):
        if new_string[i] != TEST_STRING[i]:
            diff_count += 1
    assert diff_count <= 1

    more_strings = []
    for _ in range(10):
        more_strings.append(tweaker.tweak(TEST_STRING))
    more_strings_set = set(more_strings)
    assert len(more_strings_set) > 5


def test_password_modification():
    for _ in range(100):
        seed = np.random.randint(low=0, high=1000)
        current_password = _extract_password(HAPPY_PASSWORD_STRING)
        modified_password = _modify_string(current_password, seed=seed, max_changes=1)
        assert modified_password != current_password


def test_generates_consistently_for_set_seed():
    seed = np.random.randint(low=0, high=100)
    words = ["hello", "goodbye", "banana", "apple", "yellow", "red", "green", "blue"]
    first_generation = _generate_dataset(words=words, dataset_size=6, seed=seed)
    second_generation = _generate_dataset(words=words, dataset_size=6, seed=seed)

    for i, _el in enumerate(first_generation):
        assert first_generation[i] == second_generation[i]  # list equality


def test_balanced_shuffle():
    seed = np.random.randint(low=0, high=100)
    words = [f"word-{i}" for i in range(100)]
    contexts, questions, answers = _generate_dataset(
        words=words, dataset_size=100, seed=seed
    )
    (
        _shuffled_contexts,
        _shuffled_questions,
        shuffled_answers,
    ) = _shuffle_tensor_trust_dataset(
        contexts, questions, answers, seed=seed, balanced=True
    )
    left_half_answers = shuffled_answers[:50]
    right_half_answers = shuffled_answers[50:]
    left_half_positives = np.where(np.array(left_half_answers) == "Access Granted")[0]
    left_half_negatives = np.where(np.array(left_half_answers) == "Access Denied")[0]
    right_half_positives = np.where(np.array(right_half_answers) == "Access Granted")[0]
    right_half_negatives = np.where(np.array(right_half_answers) == "Access Denied")[0]
    assert len(left_half_positives) == len(left_half_negatives)
    assert len(right_half_positives) == len(right_half_negatives)
    assert len(left_half_positives) == len(right_half_positives)
    assert len(left_half_negatives) == len(right_half_negatives)


def test_tweaks_consistently_for_set_seed():
    seed = np.random.randint(low=0, high=100)
    words = ["hello", "goodbye", "banana", "apple", "yellow", "red", "green", "blue"]
    for style in TWEAK_STYLES:
        for word in words:
            tweaker = WordTweaker(style, seed)
            first_tweak = tweaker.tweak(word)

            tweaker = WordTweaker(style, seed)
            second_tweak = tweaker.tweak(word)
            assert first_tweak == second_tweak


def test_tensor_trust_get_ground_truth_label():
    seed = 555
    environment_config = EnvironmentConfig(
        dataset_type="tensor_trust",
        train_set_size=100,
        validation_set_size=100,
        seed=seed,
    )
    tokenizer = _prepare_tokenizer(
        "bert-base-uncased", is_pythia=False, checkpoint=None
    )

    train_set, _ = get_tensor_trust_dataset(
        environment_config=environment_config,
        tokenizer=tokenizer,
        dataset_generation_style="random_words",
        seed=seed,
    )

    for example in train_set:
        assert isinstance(example, dict)
        assert tensor_trust_get_ground_truth_label(example["text"]) == example["label"]
