"""Eval niki_149 with GCG infix"""

from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

from robust_llm.batch_job_utils import run_multiple
from robust_llm.experiment_utils import get_all_n_rounds_to_evaluate

PROJECT = "AlignmentResearch/robust_llm"
BASE_MODELS = [
    "pythia-14m",
    "pythia-31m",
    "pythia-70m",
    "pythia-160m",
    "pythia-410m",
    "pythia-1b",
    "pythia-1.4b",
    "pythia-2.8b",
]


def _check_model_is_on_hfhub_impl(model_path: str, revision: str) -> bool:
    api = HfApi()
    try:
        api.repo_info(repo_id=model_path, revision=revision)
        return True
    except RepositoryNotFoundError:
        return False
    except RevisionNotFoundError:
        return False
    except Exception as e:
        print(f"An error occurred while checking: {e}")
        return False


def check_model_is_on_hfhub(model_path: str, revision: str) -> bool:
    if _check_model_is_on_hfhub_impl(model_path, revision):
        return True
    # hack like https://github.com/AlignmentResearch/robust-llm/pull/922 due to
    # inconsistent model path naming
    if model_path.startswith("AlignmentResearch/robust_llm_"):
        model_path = model_path.replace(
            "AlignmentResearch/robust_llm_",
            "AlignmentResearch/",
        )
        return _check_model_is_on_hfhub_impl(model_path, revision)
    return False


def launch(
    # name of this experiment
    experiment_name_prefix: str,
    # name of the adv training experiment
    adv_train_exp: str,
    adv_train_attack: str,
    eval_attack: str,
    dataset: str,
    # "correct seeds" in the "ICLR Adversarial training" spreadsheet
    correct_seeds=True,
    seed_range: Iterable[int] | None = None,  # deprecated
    # which seeds to eval in this launch
    seed_subrange: Iterable[int] | None = None,
    # which models to eval in this launch
    model_subrange: Iterable[int] | None = None,
    # is this a 90%-infix attack, or a suffix attack?
    is_infix_eval_attack: bool = False,
    is_prefix_eval_attack: bool = False,
    priority: str = "normal-batch",
    # make the first job high priority to check that the set of runs seems to
    # work
    is_first_job_high_priority=False,
    # use NFS? (no, we don't have checkpointing here)
    nfs: bool = False,
    cluster: str | None = None,
    check_models_exist: bool = False,
    dry_run: bool = False,
    skip_git_checks: bool = False,
    experiment_name_suffix: str = "",
):
    assert not (is_infix_eval_attack and is_prefix_eval_attack)

    # We can pack more on h100 but it seems to be worse for throughput.
    # 3-parallel 14mil takes 6 hours, 4-parallel 14mil takes 7.5 hours,
    # 7-parallel 14mil takes 17.5 hours.
    # Not sure what the sweet spot for throughput is.
    n_parallels = (
        [4, 4, 4, 4, 4, 3, 2, 1] if cluster == "h100" else [3, 3, 3, 3, 3, 2, 1, 1]
    )

    hydra_config = f"ian/iclr2024_stronger_{eval_attack}_{dataset}"
    # list of (finetune seed, adv train seed)
    seeds = [(i, i) for i in range(5)] if correct_seeds else [(0, i) for i in range(5)]
    if seed_range is not None:
        seeds = [seeds[i] for i in seed_range]
    tr_rounds_to_eval = get_all_n_rounds_to_evaluate(adv_train_attack)

    base_overrides: dict[str, str | int | float] = {}
    if eval_attack == "gcg":
        base_overrides["evaluation.num_iterations"] = 128
    elif eval_attack == "rt":
        base_overrides["evaluation.num_iterations"] = 4096
    else:
        raise ValueError(f"Unknown eval_attack: {eval_attack}")
    if is_infix_eval_attack:
        base_overrides["evaluation.evaluation_attack.perturb_position_min"] = 0.9
        base_overrides["evaluation.evaluation_attack.perturb_position_max"] = 0.9
    if is_prefix_eval_attack:
        base_overrides["evaluation.evaluation_attack.perturb_position_min"] = 0.0
        base_overrides["evaluation.evaluation_attack.perturb_position_max"] = 0.0

    overrides_and_parallelism = [
        (
            {
                **base_overrides,
                "+model": f"Default/clf/{dataset}/{base_model}-s{finetune_seed}",
                "model.name_or_path": (
                    f"{PROJECT}_clf_{dataset}_{base_model}"
                    f"_s-{finetune_seed}_adv_tr_{adv_train_attack}_t-{adv_tr_seed}"
                ),
                "model.revision": f"adv-training-round-{tr_round}",
            },
            n_parallel,
        )
        for base_model, n_parallel, tr_rounds in zip(
            BASE_MODELS, n_parallels, tr_rounds_to_eval
        )
        for tr_round in tr_rounds
        for finetune_seed, adv_tr_seed in seeds
    ]
    overrides, n_parallel = tuple(list(t) for t in zip(*overrides_and_parallelism))

    max_memory = 114 if cluster == "h100" else 52
    memories = [f"{min(max_memory, par * 17)}G" for par in n_parallel]

    priorities = [priority] * len(overrides)
    if is_first_job_high_priority:
        priorities[0] = "high-batch"

    skip_runs_mask = [False] * len(overrides)
    if seed_subrange is not None:
        seed_subrange = set(seed_subrange)
        skip_runs_mask = [
            bit or int(ov["model.name_or_path"].split("_t-")[-1]) not in seed_subrange
            for bit, ov in zip(skip_runs_mask, overrides)
        ]
    if model_subrange is not None:
        model_subrange = list(model_subrange)
        for i in model_subrange:
            assert 0 <= i < len(BASE_MODELS)
        model_set = set(BASE_MODELS[i] for i in model_subrange)
        skip_runs_mask = [
            bit or ov["+model"].split("/")[-1].split("-s")[0] not in model_set
            for bit, ov in zip(skip_runs_mask, overrides)
        ]
    if check_models_exist:
        for i, ov in enumerate(overrides):
            if skip_runs_mask[i]:
                continue
            model_path = ov["model.name_or_path"]
            model_revision = ov["model.revision"]
            if not check_model_is_on_hfhub(model_path, model_revision):
                print(f"Skipping non-existent model: {model_path}:{model_revision}")
                skip_runs_mask[i] = True

    experiment_name = f"{experiment_name_prefix}_eval_{adv_train_exp}_{eval_attack}"
    if is_infix_eval_attack:
        experiment_name += "_infix90"
    if is_prefix_eval_attack:
        experiment_name += "_prefix"
    if experiment_name_suffix:
        experiment_name += f"_{experiment_name_suffix}"

    run_multiple(
        experiment_name,
        hydra_config,
        overrides,
        n_parallel,
        memory=memories,
        priority=priorities,
        cpu=1,
        skip_runs_mask=skip_runs_mask,
        use_cluster_storage=nfs,
        cluster=cluster,
        dry_run=dry_run,
        skip_git_checks=skip_git_checks or dry_run,
    )


if __name__ == "__main__":
    path = Path(__file__)
    experiment_name_prefix = f"{path.parent.name}_{path.stem.split('_')[0][1:]}"

    launch(
        experiment_name_prefix=experiment_name_prefix,
        adv_train_exp="niki_149",
        adv_train_attack="rt",
        eval_attack="gcg",
        dataset="imdb",
        is_infix_eval_attack=True,
        correct_seeds=False,
        # let's eval just 3 seeds for now since these evals are so expensive
        seed_range=range(3),
        # jk let's just eval 1 seed at a time. seed_range is used for numbering
        # the runs since we already launched some 3-seed evals that
        # partially completed
        seed_subrange=[0],
        cluster="h100",
    )
