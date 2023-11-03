import git.repo

from robust_llm.dataset_management.tomita import make_language_generator
from robust_llm.dataset_management.tomita.tomita_base import TomitaBase

LANGUAGE_NAMES = {"tomita1", "tomita2", "tomita4", "tomita7"}


def compute_dataset_path():
    repo = git.repo.Repo(".", search_parent_directories=True)
    path_to_repo = repo.working_dir
    return f"{path_to_repo}/robust_llm/datasets/tomita"


def make_single_length_datasets(
    max_length: int = 25,
):  # length 25 leads to 1.56 GB file; could go bigger later
    for language_name in LANGUAGE_NAMES:
        t: TomitaBase = make_language_generator(language_name, max_length=5)

        for i in range(1, max_length + 1):
            print("making dataset", t, i)
            t.make_complete_dataset(i)


# Combine the datasets of all lengths up to "length" inclusive into one large dataset
def make_up_to_length_dataset(dataset_path: str, max_length: int):
    all_trues = []
    all_falses = []
    for i in range(1, max_length + 1):
        make_single_length_datasets(i)
        # Load in the dataset
        with open(f"{dataset_path}/trues_{i}.txt", "r") as f:
            trues = f.read().splitlines()
            all_trues += trues
        with open(f"{dataset_path}/falses_{i}.txt", "r") as f:
            falses = f.read().splitlines()
            all_falses += falses

    # Combine the datasets
    with open(f"{dataset_path}/trues_up_to_{max_length}.txt", "w") as f:
        f.writelines(line + "\n" for line in all_trues)
    with open(f"{dataset_path}/falses_up_to_{max_length}.txt", "w") as f:
        f.writelines(line + "\n" for line in all_falses)
    # TODO check that above still works (changed from write to writelines)


# Now make the "up to length" datasets for all the lengths and languages
def make_all_up_to_length_datasets(max_length: int):
    for language_name in LANGUAGE_NAMES:
        for i in range(1, max_length + 1):
            dataset_path = f"{compute_dataset_path()}/{language_name}"
            make_up_to_length_dataset(dataset_path, i)


# Load in a dataset of up to length "length"
def load_adversarial_dataset(language_generator_name: str, length: int):
    dataset_path = f"{compute_dataset_path()}/{language_generator_name}"
    with open(f"{dataset_path}/trues_up_to_{length}.txt", "r") as f:
        trues = f.read().splitlines()
    with open(f"{dataset_path}/falses_up_to_{length}.txt", "r") as f:
        falses = f.read().splitlines()

    # Make the dataset as a dict with "text" and "label" keys
    # Note that shuffling is managed by the Trainer
    # https://discuss.huggingface.co/t/how-to-ensure-the-dataset-is-shuffled-for-each-epoch-using-trainer-and-datasets/4212/3
    dataset = {"text": trues + falses, "label": [1] * len(trues) + [0] * len(falses)}

    return dataset


if __name__ == "__main__":
    make_all_up_to_length_datasets(
        5
    )  # this is just to test if it works, so 5 is large enough
