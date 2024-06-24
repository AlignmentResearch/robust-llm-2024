import numpy as np
import pytest


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_rng_consistency(seed):
    # We use the numpy method __setstate__ when resuming training from a checkpoint
    # so here we test that the state of the bit generator is correctly restored.
    # cf. AdversarialTrainingState.load()
    rng = np.random.default_rng(seed)
    rng.random()
    alt_rng = np.random.default_rng()
    alt_rng.bit_generator.__setstate__(rng.__getstate__())
    assert rng.random() == alt_rng.random()
    assert rng.random() == alt_rng.random()
