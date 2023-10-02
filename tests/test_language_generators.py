import pytest
from robust_llm.language_generators import make_language_generator

# 10 is long enough for all Tomita languages to have several
# true and false examples, but is otherwise arbitrary.
MAX_LANGUAGE_LENGTH = 10

TOMITA_LANGUAGES = ["Tomita1", "Tomita2", "Tomita4", "Tomita7"]

@pytest.mark.parametrize("language_name", TOMITA_LANGUAGES)
def test_generator_basic(language_name: str):
    language_generator = make_language_generator(
        language_name=language_name, max_length=MAX_LANGUAGE_LENGTH
    )

    for example in language_generator.generate_true(10):
        assert language_generator.is_in_language(
            language_generator.string_to_digit_list(example)
        )

    for example in language_generator.generate_false(10):
        assert not language_generator.is_in_language(
            language_generator.string_to_digit_list(example)
        )

@pytest.mark.parametrize("language_name", TOMITA_LANGUAGES)
@pytest.mark.parametrize("train_size", [2,10,100])
@pytest.mark.parametrize("val_size", [2,10,100])
@pytest.mark.parametrize("test_size", [0,2,10,100])
def test_generate_dataset(language_name: str, train_size: int, val_size: int, test_size: int):
    language_generator = make_language_generator(
        language_name=language_name, max_length=MAX_LANGUAGE_LENGTH
    )
    language_generator.generate_dataset(train_size=10, val_size=10, test_size=0)


# 1*
def test_tomita1_examples():
    language_generator = make_language_generator(language_name="Tomita1", max_length=MAX_LANGUAGE_LENGTH)
    assert language_generator.is_in_language([1, 1])
    assert not language_generator.is_in_language([0])


# (10)*
def test_tomita2_examples():
    language_generator = make_language_generator(language_name="Tomita2", max_length=MAX_LANGUAGE_LENGTH)
    assert language_generator.is_in_language([1, 0, 1, 0])
    assert not language_generator.is_in_language([0, 1])


# Does not contain '000'
def test_tomita4_examples():
    language_generator = make_language_generator(language_name="Tomita4", max_length=MAX_LANGUAGE_LENGTH)
    assert language_generator.is_in_language([1, 0, 1, 0])
    assert not language_generator.is_in_language([1, 0, 0, 0, 1])


# 0*1*0*1*
def test_tomita7_examples():
    language_generator = make_language_generator(language_name="Tomita7", max_length=MAX_LANGUAGE_LENGTH)
    assert language_generator.is_in_language([0])
    assert language_generator.is_in_language([0, 1, 1, 0])
    assert not language_generator.is_in_language([1, 0, 1, 0])
    assert not language_generator.is_in_language([1, 0, 1, 0, 0])
