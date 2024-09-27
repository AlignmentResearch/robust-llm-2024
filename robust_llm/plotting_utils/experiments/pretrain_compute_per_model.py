"""Estimate pretrain compute used to train each Pythia model size."""


def estimate_flops_for_training(n_params: int, n_training_tokens: int) -> int:
    """Estimate training flops using 6ND.

    This estimate is commonly used and appears in
    https://arxiv.org/pdf/2001.08361#page=9.71
    """
    return 6 * n_params * n_training_tokens


# TODO(ian): Decide how to deal with embedding params. For now I will include
# both embed and unembed.
# Sizes from scripts/ian/params_per_model.py when looking at GPTNeoXForCausalLM
# with no embeds.
size_to_n_params = {
    "14m": 7628800,
    "31m": 17616896,
    "70m": 44670976,
    "160m": 123689472,
    "410m": 353822720,
    "1b": 908759040,
    "1.4b": 1311625216,
    "2.8b": 2646430720,
    "6.9b": 6650732544,
    "12b": 11586549760,
}

# Training tokens from https://github.com/EleutherAI/pythia
n_training_tokens = 299_892_736_000

ESTIMATED_PRETRAIN_COMPUTE = {
    model_idx: estimate_flops_for_training(n_params, n_training_tokens)
    for model_idx, n_params in enumerate(size_to_n_params.values())
}
