import shutil
import sys
import tempfile
import time
from pathlib import Path

from robust_llm.config.model_configs import ModelConfig
from robust_llm.models.wrapped_model import WrappedModel


def load_model_from_disk(model_path: str, revision: str) -> WrappedModel:

    model_config = ModelConfig(
        name_or_path=model_path,
        family="pythia",
        revision=revision,
        inference_type="classification",
        env_minibatch_multiplier=1.0,
    )

    return WrappedModel.from_config(model_config, accelerator=None, num_classes=2)


def contains_necessary_files(model_path: Path) -> bool:
    necessary_files = [
        "config.json",
        "pytorch_model.bin",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    contains_nec_files = True
    for file in necessary_files:
        if not Path(model_path, file).exists():
            print(f"Missing file: {file}")
            contains_nec_files = False
    return contains_nec_files


def already_on_hfhub(model_name: str, revision: str) -> bool:
    # TODO(niki): Implement this
    return False


def load_and_push_model(
    model_name: str,
    model_path: str,
    revision: str,
) -> None:

    wrapped_model = load_model_from_disk(model_path=model_path, revision=revision)
    repo_id = f"AlignmentResearch/robust_llm_{model_name}"
    save_dir = Path(tempfile.mkdtemp())

    # Save the model and tokenizer in a temporary directory
    wrapped_model.model.save_pretrained(
        save_directory=save_dir, safe_serialization=False
    )
    wrapped_model.right_tokenizer.save_pretrained(save_directory=save_dir)

    # Upload the model
    repo_id = wrapped_model.model._create_repo(repo_id)
    wrapped_model.model._upload_modified_files(
        working_dir=save_dir,
        repo_id=repo_id,
        files_timestamps=dict(),
        commit_message="Pushing model and tokenizer to hub",
        revision=revision,  # type: ignore  # bad hinting in transformers
    )
    print("uploaded to", repo_id, ", revision", revision)

    # Clean up the temporary directory
    shutil.rmtree(save_dir)


def extract_name_and_revision(model_dir: Path) -> tuple[str, str]:
    match_string = "adv-training-round"
    parts = model_dir.name.split(match_string, 1)
    assert len(parts) == 2
    return parts[0], match_string + parts[1]


def load_and_push_from_prefix(
    task: str, attack: str, path: str = "/robust_llm_data/models"
) -> None:
    assert task in ["imdb", "spam", "pm", "wl", "helpful", "harmless"]
    assert attack in ["rt", "gcg"]

    # First we get a list of all the directories that match task and attack.
    # Models are saved in directories that look like this:
    # clf_pm_pythia-160m_s-3_adv_tr_gcg_t-3adv-training-round-33
    models_to_upload = []
    for model_dir in Path(path).iterdir():
        if task in model_dir.name and attack in model_dir.name:
            models_to_upload.append(model_dir)
    print(f"found {len(models_to_upload)} models to upload")

    # Sort the models by directory name
    models_to_upload.sort(key=lambda x: x.name)

    # Now we load and push each model.
    skipped = []
    failures = []
    retries = 5
    for i, model_dir in enumerate(models_to_upload):
        name, revision = extract_name_and_revision(model_dir)
        print(f"Uploading model {i + 1} of {len(models_to_upload)}:")
        print(f"{name}, {revision}")

        # Before trying to save, make sure that the directory
        # contains all the necessary files.
        if not contains_necessary_files(model_dir):
            print(f"Skipping {name} due to missing files.")
            skipped.append(model_dir)
            continue
        elif already_on_hfhub(name, revision):
            print(f"Skipping {name} because it is already on the hub.")
            skipped.append(model_dir)
            continue

        for attempt in range(retries):
            try:
                load_and_push_model(
                    model_name=name,
                    model_path=str(model_dir.resolve()),
                    revision=revision,
                )
                break
            except Exception as e:
                print(
                    f"Failed to push to hub on attempt {attempt + 1} of {retries}: {e}"
                )
            time.sleep(30)

            if attempt == retries - 1:
                print(f"Failed to push {model_dir} after {retries} attempts")
                failures.append(model_dir)

    if skipped:
        print(f"Skipped {len(skipped)} models:")
        for skip in skipped:
            print(skip)
    if failures:
        print(f"Failed to push {len(failures)} models:")
        for failure in failures:
            print(failure)


if __name__ == "__main__":
    task, attack = sys.argv[1], sys.argv[2]
    load_and_push_from_prefix(task, attack)
