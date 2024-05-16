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
from robust_llm.config.attack_configs import (
    BeamSearchAttackConfig,
    GCGAttackConfig,
    MultipromptGCGAttackConfig,
    SearchBasedAttackConfig,
)
from robust_llm.models import WrappedModel


def make_runner(
    wrapped_model: WrappedModel,
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

    # This match-case statement uses class patterns, as described in this SO
    # answer: https://stackoverflow.com/a/67524642
    match config:
        case BeamSearchAttackConfig():
            return BeamSearchRunner(
                **base_args,  # type: ignore
                beam_search_width=config.beam_search_width,
            )

        case GCGAttackConfig():
            return GCGRunner(
                **base_args,  # type: ignore
                top_k=config.top_k,
            )
        case MultipromptGCGAttackConfig():
            return MultiPromptGCGRunner(
                **base_args,  # type: ignore
                top_k=config.top_k,
            )
        case _:
            raise ValueError(f"Unknown SearchBasedAttackConfig type for: {config}")
