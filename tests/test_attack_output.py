from unittest.mock import MagicMock

import pytest

from robust_llm.attacks.attack import AttackOutput


def test_wrong_len():
    dataset = MagicMock()
    dataset.ds = [1, 2, 3]
    per_example_info = {"key": [1, 2]}
    # Check that the error message is informative.
    with pytest.raises(ValueError) as excinfo:
        AttackOutput(
            dataset=dataset,
            attack_data=None,
            per_example_info=per_example_info,
            flops=0,
        )
    assert "Length of per_example_info[key] (2)" in str(excinfo.value)
