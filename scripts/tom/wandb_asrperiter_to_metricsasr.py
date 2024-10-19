"""Updates adversarial_eval/asr_per_iteration -> metrics/asr@i in W&B runs.

Updates runs that have adversarial_eval/asr_per_iteration data to have
metrics/asr@i data, because adversarial_eval/asr_per_iteration data is annoying
to parse due to it being a list. Assumes asr_per_iteration is a list of 11 values.
"""

import time

import wandb

api = wandb.Api()

groups = [
    "ian_102a_gcg_pythia_harmless",
    "ian_103a_gcg_pythia_helpful",
    "ian_106_gcg_pythia_imdb",
    "ian_107_gcg_pythia_pm",
    "ian_108_gcg_pythia_wl",
    "ian_109_gcg_pythia_spam",
]

for group in groups:
    runs = api.runs(
        path="farai/robust-llm",
        filters={"group": group, "state": "finished"},
    )
    for run in runs:
        asr_per_iteration = run.summary.get("adversarial_eval/asr_per_iteration")

        if (
            asr_per_iteration is None
            or not isinstance(asr_per_iteration, list)
            or len(asr_per_iteration) != 11
        ):
            print(
                f"Skipping run '{run.name}':"
                " Invalid or missing 'adversarial_eval/asr_per_iteration' data"
            )
            continue

        if f"metrics/asr@{len(asr_per_iteration) - 1}" in run.summary:
            print(f"Skipping run '{run.name}': Metrics already exist")
            continue

        for i, asr_value in enumerate(asr_per_iteration):
            run.summary[f"metrics/asr@{i}"] = asr_value

        for i in range(3):
            try:
                run.summary.update()
                print(f"Updated run '{run.name}'")
                break
            except Exception as e:
                print(f"Failed to update run '{run.name}': {e}")
                if i < 2:
                    sleep_time = 15 if i == 0 else 60
                    print(f"Waiting for {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    print("Failed, moving on to the next run")
