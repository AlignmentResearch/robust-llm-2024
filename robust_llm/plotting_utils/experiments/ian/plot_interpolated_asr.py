import matplotlib.pyplot as plt

from robust_llm.metrics.asr_per_iteration import (
    ASRMetricResults,
    compute_asr_per_iteration_from_wandb,
)


def plot_interpolated_asr_from_asrs(asrs: ASRMetricResults) -> None:
    """Computes the interpolated ASR from the ASRs for each iteration of the attack."""
    plt.plot(asrs.asr_per_iteration, label="raw", linewidth=4)
    # plt.xscale("log")
    # plt.yscale("log")
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
    group_name = "ian_106_gcg_pythia_imdb"
    # run_index = "0009"
    run_index = "0000"
    asrs = compute_asr_per_iteration_from_wandb(group_name, run_index)
    plot_interpolated_asr_from_asrs(asrs)
