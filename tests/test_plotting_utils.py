import matplotlib.pyplot as plt
import pandas as pd
import pytest

from robust_llm.plotting_utils.tools import (
    _get_num_params_from_name,
    _get_pretraining_fraction,
    create_legend,
    extract_size_from_model_name,
    get_color_palette,
    get_legend_handles,
    postprocess_data,
    set_up_paper_plot,
)
from robust_llm.wandb_utils.constants import FINAL_PYTHIA_CHECKPOINT


def test_set_up_paper_plot():
    fig, ax = plt.subplots()
    set_up_paper_plot(fig, ax)

    assert fig.get_size_inches().tolist() == [3.5, 2.5]
    assert ax.get_gridspec()
    assert ax.get_axisbelow()

    # Check tick parameters
    major_ticks = ax.get_xticks()
    minor_ticks = ax.get_xticks(minor=True)
    assert len(major_ticks) > 0
    assert len(minor_ticks) > 0


def test_get_color_palette():
    data = pd.DataFrame(
        {"num_params": [1e6, 1e7, 1e8], "other_column": ["A", "B", "C"]}
    )

    palette_dict = get_color_palette(data, "num_params")
    assert len(palette_dict) == 3
    assert all(isinstance(color, tuple) for color in palette_dict.values())

    palette_dict = get_color_palette(data, "other_column")
    assert len(palette_dict) == 3
    assert all(isinstance(color, tuple) for color in palette_dict.values())


def test_get_legend_handles():
    data = pd.DataFrame(
        {"num_params": [1e6, 1e7, 1e8], "adv_training_round": [1, 2, 3]}
    )
    palette_dict = {1e6: (1, 0, 0), 1e7: (0, 1, 0), 1e8: (0, 0, 1)}

    legend_handles = get_legend_handles(data, "num_params", palette_dict)
    assert len(legend_handles) == 3
    assert all(
        isinstance(handle, plt.Line2D)  # type: ignore
        for handle in legend_handles.values()
    )

    palette_dict = {1: (1, 0, 0), 2: (0, 1, 0), 3: (0, 0, 1)}
    legend_handles = get_legend_handles(data, "adv_training_round", palette_dict)
    assert len(legend_handles) == 3
    assert all(
        isinstance(handle, plt.Line2D)  # type: ignore
        for handle in legend_handles.values()
    )


def test_create_legend():
    fig, ax = plt.subplots()
    legend_handles = {
        1e6: plt.Line2D([0], [0], color="red", label="1M"),  # type: ignore
        1e7: plt.Line2D([0], [0], color="green", label="10M"),  # type: ignore
        1e8: plt.Line2D([0], [0], color="blue", label="100M"),  # type: ignore
    }

    create_legend("num_params", ax, legend_handles)
    legend = ax.get_legend()
    assert legend is not None
    assert "param" in legend.get_title().get_text().lower()

    with pytest.raises(ValueError):
        create_legend("unsupported_column", ax, legend_handles)


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "model_name_or_path": ["pythia-14m", "pythia-6.9b"],
            "revision": ["adv-training-round-0", "adv-training-round-1"],
            "metric_1": [0.1, 0.2],
            "metric_2": [0.3, 0.4],
            "_step": [1, 2],
            "seed_idx": [1, 2],
        }
    )


def test_postprocess_data(sample_data):
    postprocess_data(sample_data)
    assert "num_params" in sample_data.columns
    assert "pretraining_fraction" in sample_data.columns


def test_get_num_params_from_name():
    assert _get_num_params_from_name("pythia-70m") == 44_672_000
    with pytest.raises(AssertionError):
        _get_num_params_from_name("invalid_model_name")


def test_get_pretraining_fraction():
    assert _get_pretraining_fraction("model-ch-1000") == 1000 / FINAL_PYTHIA_CHECKPOINT
    assert _get_pretraining_fraction("model-no-checkpoint") == 1.0


def test_extract_size_from_model_name():
    assert extract_size_from_model_name("Qwen/Qwen1.5-0.5B-Chat") == 500_000_000
    assert extract_size_from_model_name("Qwen/Qwen1.5-1.8B-Chat") == 1_800_000_000
    assert extract_size_from_model_name("meta-llama/Llama-2-7b-hf") == 7_000_000_000
    assert extract_size_from_model_name("EleutherAI/pythia-14m") == 14_000_000
