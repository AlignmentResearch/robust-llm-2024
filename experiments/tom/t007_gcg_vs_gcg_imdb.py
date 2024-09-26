"""Eval niki_152 with GCG"""

from pathlib import Path
from typing import Iterable

from t005b_gcg_infix_vs_rt_imdb import (
    BASE_MODELS,
    PROJECT,
    check_model_is_on_hfhub,
    get_tr_rounds_to_eval,
    launch,
)


def check_model_seeds(
    adv_train_attack: str,
    dataset: str,
    model_subrange: Iterable[int] | None = None,
) -> tuple[list[tuple[str, int, int]], list[tuple[str, int]]]:
    # Returns:
    # - (model, adv round, existing seed) for each model with at least one
    #   existing seed
    # - (model, adv round) for each model missing all seeds
    existing_models = []
    missing_models = []
    tr_rounds_to_eval = get_tr_rounds_to_eval(
        max_adv_tr_rounds=250 if adv_train_attack == "rt" else 60
    )
    models_and_rounds: Iterable[tuple[str, list[int]]] = zip(
        BASE_MODELS, tr_rounds_to_eval
    )
    if model_subrange is not None:
        models_and_rounds = list(models_and_rounds)
        models_and_rounds = [models_and_rounds[i] for i in model_subrange]
    for model, rounds in models_and_rounds:
        for r in rounds:
            exists = False
            for seed in range(5):
                model_name = (
                    f"{PROJECT}_clf_{dataset}_{model}_s-{seed}"
                    f"_adv_tr_{adv_train_attack}_t-{seed}"
                )
                revision = f"adv-training-round-{r}"
                if check_model_is_on_hfhub(model_name, revision):
                    existing_models.append((model, r, seed))
                    exists = True
                    break
            if not exists:
                missing_models.append((model, r))
    return existing_models, missing_models


if __name__ == "__main__":
    path = Path(__file__)
    experiment_name_prefix = f"{path.parent.name}_{path.stem.split('_')[0][1:]}"

    # # print out niki_152a missing models
    # existing_models, missing_models = check_model_seeds(
    #     adv_train_attack="gcg",
    #     dataset="imdb",
    #     model_subrange=range(0, 4),
    # )
    # print("EXISTING")
    # for em in existing_models:
    #     print(em)
    # print("MISSING")
    # for m in missing_models:
    #     print(mm)
    # assert False

    # models split in two because they're split across niki_152 and niki_152a
    launch(
        experiment_name_prefix=experiment_name_prefix,
        adv_train_exp="niki_152",
        adv_train_attack="gcg",
        eval_attack="gcg",
        dataset="imdb",
        is_infix_eval_attack=False,
        correct_seeds=False,
        # eval one seed at a time
        seed_subrange=[0],
        model_subrange=range(4, 8),
        cluster="h100",
        skip_git_checks=False,
    )

    launch(
        experiment_name_prefix=experiment_name_prefix,
        # maybe should rename this to niki_152 to match above, since this is
        # only used for adv train name? I think I'll still keep it different
        # since niki may rerun niki_152 to use correct seeds
        adv_train_exp="niki_152a",
        adv_train_attack="gcg",
        eval_attack="gcg",
        dataset="imdb",
        is_infix_eval_attack=False,
        correct_seeds=True,
        # eval one seed at a time
        seed_subrange=[0],
        model_subrange=range(0, 4),
        cluster="a6k",
        skip_git_checks=True,
        is_first_job_high_priority=False,
    )
