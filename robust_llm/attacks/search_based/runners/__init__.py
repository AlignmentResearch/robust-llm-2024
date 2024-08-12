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
    victim: WrappedModel,
    prepped_examples: list[PreppedExample],
    random_seed: int,
    n_its: int,
    config: SearchBasedAttackConfig,
) -> SearchBasedRunner | MultiPromptSearchBasedRunner:
    base_args = {
        "victim": victim,
        "n_candidates_per_it": config.n_candidates_per_it,
        "n_its": n_its,
        "n_attack_tokens": config.n_attack_tokens,
        "scores_from_text_callback": config.scores_from_text_callback,
        "prepped_examples": prepped_examples,
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
                differentiable_embeds_callback=config.differentiable_embeds_callback,
                top_k=config.top_k,
            )
        case MultipromptGCGAttackConfig():
            return MultiPromptGCGRunner(
                **base_args,  # type: ignore
                differentiable_embeds_callback=config.differentiable_embeds_callback,
                top_k=config.top_k,
            )
        case _:
            raise ValueError(f"Unknown SearchBasedAttackConfig type for: {config}")
