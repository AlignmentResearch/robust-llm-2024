import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
# Just take the ian_xxx part as the version name, but use a hyphen to
# fit the convention of _ for separation.
DATASET = "imdb"
VERSION = "ian-079"
HYDRA_CONFIG = f"ian/gcg_{DATASET}"


BASE_MODEL_GPU_MEMORY: list[tuple[str, str, int, str]] = [
    (
        "EleutherAI/pythia-14m",
        "AlignmentResearch/robust_llm_pythia-14m_clf_{dataset}_v-{version}_s-{seed}",
        1,
        "10G",
    ),
    (
        "EleutherAI/pythia-31m",
        "AlignmentResearch/robust_llm_pythia-31m_clf_{dataset}_v-{version}_s-{seed}",
        1,
        "20G",
    ),
    (
        "EleutherAI/pythia-70m",
        "AlignmentResearch/robust_llm_pythia-70m_clf_{dataset}_v-{version}_s-{seed}",
        1,
        "20G",
    ),
    (
        "EleutherAI/pythia-160m",
        "AlignmentResearch/robust_llm_pythia-160m_clf_{dataset}_v-{version}_s-{seed}",
        1,
        "20G",
    ),
    (
        "EleutherAI/pythia-410m",
        "AlignmentResearch/robust_llm_pythia-410m_clf_{dataset}_v-{version}_s-{seed}",
        1,
        "30G",
    ),
    (
        "EleutherAI/pythia-1b",
        "AlignmentResearch/robust_llm_pythia-1b_clf_{dataset}_v-{version}_s-{seed}",
        1,
        "30G",
    ),
    (
        "EleutherAI/pythia-1.4b",
        "AlignmentResearch/robust_llm_pythia-1.4b_clf_{dataset}_v-{version}_s-{seed}",
        1,
        "40G",
    ),
    (
        "EleutherAI/pythia-2.8b",
        "AlignmentResearch/robust_llm_pythia-2.8b_clf_{dataset}_v-{version}_s-{seed}",
        2,
        "40G",
    ),
    (
        "EleutherAI/pythia-6.9b",
        "AlignmentResearch/robust_llm_pythia-6.9b_clf_{dataset}_v-{version}_s-{seed}",
        2,
        "80G",
    ),
    (
        "EleutherAI/pythia-12b",
        "AlignmentResearch/robust_llm_pythia-12b_clf_{dataset}_v-{version}_s-{seed}",
        2,
        "100G",
    ),
]

SEEDS = [0, 1, 2, 3, 4]

OVERRIDE_TUPLES = [
    (
        {
            "environment.deterministic": True,
            "+model": base,
            "model.name_or_path": model.format(
                dataset=DATASET, version=VERSION, seed=seed
            ),
        },
        n_gpus,
        memory,
    )
    for seed in SEEDS
    for (base, model, n_gpus, memory) in BASE_MODEL_GPU_MEMORY
]

OVERRIDE_ARGS_LIST = [x[0] for x in OVERRIDE_TUPLES]
N_GPUS = [x[1] for x in OVERRIDE_TUPLES]
MEMORY = [x[2] for x in OVERRIDE_TUPLES]

if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        gpu=N_GPUS,
        memory=MEMORY,
        cpu=8,
        priority="high-batch",
    )
