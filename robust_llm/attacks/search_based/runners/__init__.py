from robust_llm.attacks.search_based.models import SearchBasedAttackWrappedModel
from robust_llm.attacks.search_based.runners.beam_search_runner import BeamSearchRunner
from robust_llm.attacks.search_based.runners.gcg_runner import GCGRunner
from robust_llm.attacks.search_based.runners.search_based_runner import (
    SearchBasedRunner,
)
from robust_llm.attacks.search_based.utils import PromptTemplate
from robust_llm.configs import SearchBasedAttackConfig


def make_runner(
    wrapped_model: SearchBasedAttackWrappedModel,
    prompt_template: PromptTemplate,
    clf_target: int,
    random_seed: int,
    config: SearchBasedAttackConfig,
) -> SearchBasedRunner:
    base_args = {
        "wrapped_model": wrapped_model,
        "n_candidates_per_it": config.n_candidates_per_it,
        "n_its": config.n_its,
        "n_attack_tokens": config.n_attack_tokens,
        "forward_pass_batch_size": config.forward_pass_batch_size,
        "prompt_template": prompt_template,
        "seq_clf": config.seq_clf,
        "clf_target": clf_target,
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
    else:
        raise ValueError(f"Unknown search type: {search_type}")
