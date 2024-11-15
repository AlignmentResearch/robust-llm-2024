import json
import os
import tempfile
from dataclasses import fields
from pathlib import Path

import pytest

from robust_llm.attacks.attack import AttackState


@pytest.fixture
def complex_attack_state():
    """Create an AttackState with various special characters and edge cases."""
    return AttackState(
        rng_state={"numpy": [1, 2, 3], "torch": [4, 5, 6]},
        example_index=42,
        attacked_texts=[
            "Normal text",
            'Text with "quotes"',
            "Text with 'single quotes'",
            "Text with \n newlines",
            "Text with Unicode: 你好, ß, 🌟",
            "Text with LaTeX: $\\alpha + \\beta$",
            'Text with JSON chars: {}, [], "',
            "Text with backslashes: C:\\path\\to\\file",
            "Text with control chars: \t\r\n",
            """Multi
            line
            text""",
        ],
        all_iteration_texts=[
            ["First iter", 'Second "iter"'],
            ["Unicode iter: 经济", "LaTeX iter: $\\frac{1}{2}$"],
        ],
        attacks_info={
            "scores": [[1.0, 2.0], [3.0, 4.0]],
            "metadata": ["test1", "test2"],
        },
    )


def test_save_and_load_roundtrip(complex_attack_state):
    """Test that saving and loading preserves all data exactly."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        path = Path(tf.name)

    try:
        # Save
        complex_attack_state.save(path)

        # Load
        loaded_state = AttackState.load(path)

        # Get all fields from the dataclass
        # Compare each field
        for field in fields(complex_attack_state):
            original_value = getattr(complex_attack_state, field.name)
            loaded_value = getattr(loaded_state, field.name)

            assert original_value == loaded_value

    finally:
        os.unlink(path)


def test_special_characters_handling(complex_attack_state):
    """Test that special characters are preserved correctly."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        path = Path(tf.name)

    try:
        complex_attack_state.save(path)
        loaded_state = AttackState.load(path)

        # Test specific special characters
        assert "你好" in loaded_state.attacked_texts[4]
        assert "\\alpha" in loaded_state.attacked_texts[5]
        assert "\n" in loaded_state.attacked_texts[3]
        assert "\\path\\to" in loaded_state.attacked_texts[7]
        assert "\t\r\n" in loaded_state.attacked_texts[8]
    finally:
        os.unlink(path)


def test_json_file_readability():
    """Test that the generated JSON file is human-readable."""
    state = AttackState(
        rng_state={"test": [1, 2]},
        attacked_texts=['Test "quote"', "Test 你好"],
        all_iteration_texts=[["Test1", "Test2"]],
    )

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        path = Path(tf.name)

    try:
        state.save(path)

        # Read raw file contents
        with open(path, encoding="utf-8") as f:
            content = f.read()

        # Check formatting
        assert content.count("\n") > 5  # Should be properly indented
        assert '"attacked_texts"' in content  # Keys should be readable
        assert "你好" in content  # Unicode should be preserved
    finally:
        os.unlink(path)


def test_empty_state():
    """Test handling of empty state."""
    empty_state = AttackState(rng_state={})
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        path = Path(tf.name)

    try:
        empty_state.save(path)
        loaded_state = AttackState.load(path)

        assert loaded_state.attacked_texts == []
        assert loaded_state.all_iteration_texts == []
        assert loaded_state.attacks_info == {}
    finally:
        os.unlink(path)


def test_invalid_json():
    """Test handling of invalid JSON file."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        tf.write(b'{"invalid": json')
        path = Path(tf.name)

    try:
        with pytest.raises(json.JSONDecodeError):
            AttackState.load(path)
    finally:
        os.unlink(path)
