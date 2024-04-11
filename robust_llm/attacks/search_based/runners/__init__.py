from robust_llm.attacks.search_based.models import SearchBasedAttackWrappedModel
from robust_llm.attacks.search_based.runners.beam_search_runner import BeamSearchRunner
from robust_llm.attacks.search_based.runners.gcg_runner import GCGRunner
from robust_llm.attacks.search_based.runners.multiprompt_gcg_runner import (
    MultiPromptGCGRunner,
)
from robust_llm.attacks.search_based.runners.multiprompt_search_based_runner import (
    MultiPromptSearchBasedRunner,
)
from robust_llm.attacks.search_based.runners.search_based_runner import (
    SearchBasedRunner,
)
from robust_llm.attacks.search_based.utils import PreppedExample
from robust_llm.configs import SearchBasedAttackConfig


def make_runner(
    wrapped_model: SearchBasedAttackWrappedModel,
    prepped_examples: list[PreppedExample],
    random_seed: int,
    config: SearchBasedAttackConfig,
) -> SearchBasedRunner | MultiPromptSearchBasedRunner:
    base_args = {
        "wrapped_model": wrapped_model,
        "n_candidates_per_it": config.n_candidates_per_it,
        "n_its": config.n_its,
        "n_attack_tokens": config.n_attack_tokens,
        "forward_pass_batch_size": config.forward_pass_batch_size,
        "prepped_examples": prepped_examples,
        "seq_clf": config.seq_clf,
        "random_seed": random_seed,
    }

    search_type = config.search_type

    if search_type == "beam_search":
        return BeamSearchRunner(
            **base_args,  # type: ignore
            beam_search_width=config.beam_search_attack_config.beam_search_width,
        )

    elif search_type == "gcg":
        return GCGRunner(
            **base_args,  # type: ignore
            top_k=config.gcg_attack_config.top_k,
        )
    elif search_type == "multiprompt_gcg":
        return MultiPromptGCGRunner(
            **base_args,  # type: ignore
            top_k=config.gcg_attack_config.top_k,
        )
    else:
        raise ValueError(f"Unknown search type: {search_type}")
