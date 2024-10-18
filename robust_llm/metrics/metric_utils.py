import time

import datasets
from wandb.apis.public.runs import Run as WandbRun

from robust_llm.attacks.attack import AttackedRawInputOutput, AttackOutput
from robust_llm.rllm_datasets.load_rllm_dataset import load_rllm_dataset
from robust_llm.wandb_utils.wandb_api_tools import (
    get_attack_data_tables,
    get_dataset_config_from_run,
    get_wandb_run,
)


def get_attack_output_from_wandb(
    group_name: str, run_index: str, max_workers: int = 4
) -> AttackOutput:
    run = get_wandb_run(group_name, run_index)
    return get_attack_output_from_wandb_run(run, max_workers=max_workers)


def get_attack_output_from_wandb_run(
    run: WandbRun, max_workers: int = 2
) -> AttackOutput:
    dataset_cfg = get_dataset_config_from_run(run)

    tic = time.perf_counter()
    dataset = load_rllm_dataset(dataset_cfg, split="validation")
    toc = time.perf_counter()
    print(f"Loaded dataset in {toc - tic:.2f} seconds")

    tic = time.perf_counter()
    attack_data_dfs = get_attack_data_tables(run, max_workers=max_workers)
    dataset_indices = list(attack_data_dfs.keys())
    toc = time.perf_counter()
    print(f"Loaded attack data for {run.name} in {toc - tic:.2f} seconds")

    # We only want the ones that were actually attacked
    dataset = dataset.get_subset(dataset_indices)

    attack_data = AttackedRawInputOutput.from_dfs(attack_data_dfs)
    attack_output = AttackOutput(dataset=dataset, attack_data=attack_data)
    return attack_output


def _dataset_for_iteration(
    attack_out: AttackOutput, iteration: int
) -> datasets.Dataset:
    ds = attack_out.dataset.ds
    # TODO(ian): Remove these asserts
    assert attack_out.attack_data is not None
    all_iteration_texts = attack_out.attack_data.iteration_texts
    texts = [all_iteration_texts[i][iteration] for i in range(len(all_iteration_texts))]
    clf_labels = ds["clf_label"]
    gen_targets = ds["gen_target"]
    proxy_clf_labels = ds["proxy_clf_label"]
    proxy_gen_targets = ds["proxy_gen_target"]

    ds = datasets.Dataset.from_dict(
        {
            "text": texts,
            "clf_label": clf_labels,
            "gen_target": gen_targets,
            "proxy_clf_label": proxy_clf_labels,
            "proxy_gen_target": proxy_gen_targets,
        }
    )
    return ds


def _compute_clf_asr_from_logits(logits: list[list[float]], labels: list[int]):
    """If we have cached logits then use those to compute the ASR."""
    n_examples = len(labels)
    n_correct = 0
    for i in range(n_examples):
        pred = max(range(len(logits[i])), key=lambda x: logits[i][x])
        if pred == labels[i]:
            n_correct += 1
    return 1 - n_correct / n_examples
