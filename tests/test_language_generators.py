import pytest
from robust_llm.language_generators import make_language_generator


@pytest.mark.parametrize("language_name", ["Tomita1", "Tomita2", "Tomita4", "Tomita7"])
def test_generator_basic(language_name: str):
    language_generator = make_language_generator(
        language_name=language_name, max_length=10
    )

    for example in language_generator.generate_true(3):
        assert language_generator.is_in_language(
            language_generator.string_to_digit_list(example)
        )

    for example in language_generator.generate_false(3):
        assert not language_generator.is_in_language(
            language_generator.string_to_digit_list(example)
        )


# 1*
def test_tomita1_examples():
    language_generator = make_language_generator(language_name="Tomita1", max_length=10)
    assert language_generator.is_in_language([1, 1])
    assert not language_generator.is_in_language([0])


# (10)*
def test_tomita2_examples():
    language_generator = make_language_generator(language_name="Tomita2", max_length=10)
    assert language_generator.is_in_language([1, 0, 1, 0])
    assert not language_generator.is_in_language([0, 1])


# Does not contain '000'
def test_tomita4_examples():
    language_generator = make_language_generator(language_name="Tomita4", max_length=10)
    assert language_generator.is_in_language([1, 0, 1, 0])
    assert not language_generator.is_in_language([1, 0, 0, 0, 1])


# 0*1*0*1*
def test_tomita7_examples():
    language_generator = make_language_generator(language_name="Tomita7", max_length=10)
    assert language_generator.is_in_language([0])
    assert language_generator.is_in_language([0, 1, 1, 0])
    assert not language_generator.is_in_language([1, 0, 1, 0])
    assert not language_generator.is_in_language([1, 0, 1, 0, 0])
