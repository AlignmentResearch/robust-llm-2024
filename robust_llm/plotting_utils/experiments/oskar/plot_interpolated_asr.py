import argparse

import matplotlib.pyplot as plt

from robust_llm.attacks.attack import AttackedRawInputOutput, AttackOutput
from robust_llm.metrics.asr_per_iteration import (
    ASRMetricResults,
    compute_asr_per_iteration_from_logits,
)
from robust_llm.rllm_datasets.load_rllm_dataset import load_rllm_dataset
from robust_llm.wandb_utils.wandb_api_tools import (
    _maybe_get_attack_data_from_artifacts,
    _maybe_get_attack_data_from_storage,
    get_dataset_config_from_run,
    get_run_from_index,
)


def plot_interpolated_asr_from_asrs(asrs: ASRMetricResults) -> None:
    """Computes the interpolated ASR from the ASRs for each iteration of the attack."""
    plt.plot(asrs.asr_per_iteration, label="raw", linewidth=4)
    print(asrs.asr_per_iteration)
    deciles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    interpolated_iterations = [
        asrs.interpolated_iteration_for_asr(decile) for decile in deciles
    ]
    # filter out Nones
    it_decile_pairs = list(
        filter(lambda x: x[1] is not None, zip(interpolated_iterations, deciles))
    )
    filtered_its, filtered_deciles = zip(*it_decile_pairs)
    print(filtered_its, filtered_deciles)
    plt.plot(
        filtered_its, filtered_deciles, marker="o", color="red", label="interpolated"
    )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot interpolated ASR from ASRs.")
    parser.add_argument(
        "--group_name", type=str, help="The group name for the experiment."
    )
    parser.add_argument(
        "--run_index", type=str, help="The run index for the experiment."
    )

    args = parser.parse_args()
    run = get_run_from_index(args.group_name, args.run_index)
    wandb_run = run.to_wandb()
    dataset_cfg = get_dataset_config_from_run(run)
    wandb_run = run.to_wandb()

    for method in (
        _maybe_get_attack_data_from_storage,
        _maybe_get_attack_data_from_artifacts,
    ):
        attack_data_dfs = method(wandb_run)
        assert attack_data_dfs is not None
        dataset_indices = list(attack_data_dfs.keys())

        # We only want the ones that were actually attacked
        dataset = load_rllm_dataset(dataset_cfg, split="validation")
        dataset = dataset.get_subset(dataset_indices)

        attack_data = AttackedRawInputOutput.from_dfs(attack_data_dfs)
        attack_output = AttackOutput(dataset=dataset, attack_data=attack_data, flops=0)

        asrs = compute_asr_per_iteration_from_logits(attack_output)
        plot_interpolated_asr_from_asrs(asrs)
