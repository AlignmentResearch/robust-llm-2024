# For ease of plotting in the wandb dashboard, adds extra fields to wandb runs
# doing evals of adversarially trained models.
# - Parses adv training round out of model.revision
# - Parses base model out of our adv training model name
import re
import time

import wandb

api = wandb.Api()

groups = [
    # "tom_004a_gcg_prefix_imdb",
    # "tom_004b_gcg_90pc_imdb",
    "tom_005a_eval_niki_149_gcg",
    "tom_006_eval_niki_150_gcg",
]


def extract_round(revision: str) -> int:
    """Extracts integer from "adv-training-round-<integer>"""
    match = re.search(r"adv-training-round-(\d+)", revision)
    assert match
    return int(match.group(1))


def extract_base_model(name_or_path: str) -> str:
    # name_or_path looks like
    # AlignmentResearch/robust_llm_pythia-410m_niki-052_imdb_gcg_seed-0
    match = re.search(r"(pythia-[^_]+)_", name_or_path)
    assert match
    return match.group(1)


for group in groups:
    runs = api.runs(
        path="farai/robust-llm",
        filters={"group": group, "state": "finished"},
    )
    for run in runs:
        if "adv_training_round" in run.summary and "base_model" in run.summary:
            continue

        revision = run.summary["experiment_yaml"]["model"]["revision"]
        round_number = extract_round(revision)
        run.summary["adv_training_round"] = round_number

        model = run.summary["experiment_yaml"]["model"]["name_or_path"]
        base_model = extract_base_model(model)
        run.summary["base_model"] = base_model

        for _ in range(3):
            try:
                run.summary.update()
                print(f"Updated run '{run.name}'")
                break
            except Exception as e:
                print(f"Failed to update run '{run.name}': {e}")
                print("Waiting for 15 seconds...")
                time.sleep(15)
