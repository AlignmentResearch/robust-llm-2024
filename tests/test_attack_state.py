import random
import shutil
from pathlib import Path

import pytest
import torch
from accelerate import Accelerator

from robust_llm.attacks.attack import AttackedExample, AttackState
from robust_llm.dist_utils import DistributedRNG
from robust_llm.state_classes.rng_state import RNGState


@pytest.fixture
def complex_attack_state():
    """Create an AttackState with various special characters and edge cases."""
    examples = []
    attacked_texts = [
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
    ]

    n_its = 5
    for i, attacked_text in enumerate(attacked_texts):
        examples.append(
            AttackedExample(
                example_index=i,
                attacked_text=attacked_text,
                iteration_texts=[attacked_text] * n_its,
                logits=[[random.random() for _ in range(4)] for _ in range(n_its)],
                flops=1,
            )
        )

    dist_rng = DistributedRNG(seed=42, accelerator=None)
    rng_state = RNGState(
        torch_rng_state=torch.random.get_rng_state(),
        distributed_rng=dist_rng,
    )

    attack_state = AttackState(
        previously_attacked_examples=tuple(examples),
        rng_state=rng_state,
    )
    return attack_state


def test_save_and_load_roundtrip(complex_attack_state: AttackState):
    """Test that saving and loading preserves all data exactly."""
    accelerator = Accelerator()
    path = Path("/tmp/robust-llm/tests/save_and_load_roundtrip")
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=False)
    try:
        # Save
        complex_attack_state.save(path, accelerator.process_index, accelerator)

        # Load
        path = path / "example_00010"
        loaded_state = AttackState.load(path, accelerator.process_index, accelerator)

        original_examples = complex_attack_state.previously_attacked_examples
        reloaded_examples = loaded_state.previously_attacked_examples
        assert original_examples == reloaded_examples

        original_rng_state = complex_attack_state.rng_state
        reloaded_rng_state = loaded_state.rng_state
        assert torch.equal(
            original_rng_state.torch_rng_state, reloaded_rng_state.torch_rng_state
        )
        assert (
            original_rng_state.distributed_rng.getstate()
            == reloaded_rng_state.distributed_rng.getstate()
        )

    finally:
        shutil.rmtree(path)


def test_special_characters_handling(complex_attack_state: AttackState):
    """Test that special characters are preserved correctly."""
    accelerator = Accelerator()
    path = Path("/tmp/robust-llm/tests/special_characters_handling")
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=False)

    try:
        # Save
        complex_attack_state.save(path, accelerator.process_index, accelerator)

        # Load
        path = path / "example_00010"
        loaded_state = AttackState.load(path, accelerator.process_index, accelerator)

        # Test specific special characters
        assert "你好" in loaded_state.previously_attacked_examples[4].attacked_text
        assert "\\alpha" in loaded_state.previously_attacked_examples[5].attacked_text
        assert "\n" in loaded_state.previously_attacked_examples[3].attacked_text
        assert (
            "\\path\\to" in loaded_state.previously_attacked_examples[7].attacked_text
        )
        assert "\t\r\n" in loaded_state.previously_attacked_examples[8].attacked_text
    finally:
        shutil.rmtree(path)
