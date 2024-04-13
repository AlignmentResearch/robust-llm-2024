import pytest

from robust_llm.utils import BalancedSampler


@pytest.mark.parametrize("regular_data_size", [1, 2, 3])
@pytest.mark.parametrize("adversarial_data_size", [1, 2, 3])
def test_balanced_sampler(regular_data_size: int, adversarial_data_size: int):
    regular_data = [0] * regular_data_size
    adversarial_data = [1] * adversarial_data_size
    balanced_sampler = BalancedSampler(regular_data, adversarial_data)

    assert len(balanced_sampler) == 2 * regular_data_size

    indices_for_regular_data = []

    for i, idx in enumerate(balanced_sampler):
        if i % 2 == 0:
            assert idx < regular_data_size
            indices_for_regular_data.append(idx)
        else:
            assert regular_data_size <= idx < regular_data_size + adversarial_data_size

    # Ensure all regular data points were sampled exactly once
    assert list(sorted(indices_for_regular_data)) == list(range(regular_data_size))
