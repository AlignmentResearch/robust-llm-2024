from robust_llm.language_generators.tomita1 import Tomita1
from robust_llm.language_generators.tomita2 import Tomita2
from robust_llm.language_generators.tomita4 import Tomita4
from robust_llm.language_generators.tomita7 import Tomita7


language_generators = {Tomita1, Tomita2, Tomita4, Tomita7}
DATASET_PATH = "/home/dev/robust-llm/robust_llm/datasets"


def make_single_length_datasets():
    for language_generator in language_generators:
        t = language_generator(max_length=50)

        for i in range(1, 26):
            print("making dataset", t, i)
            t.make_complete_dataset(i)


# Combine the datasets of all lengths up to "length" inclusive into on large dataset
def make_up_to_length_dataset(dataset_path, length):
    all_trues = []
    all_falses = []
    for i in range(1, length + 1):
        # Load in the dataset
        with open(f"{dataset_path}/trues_{i}.txt", "r") as f:
            trues = f.read().splitlines()
            all_trues += trues
        with open(f"{dataset_path}/falses_{i}.txt", "r") as f:
            falses = f.read().splitlines()
            all_falses += falses

    # Combine the datasets
    with open(f"{dataset_path}/trues_up_to_{i}.txt", "w") as f:
        f.write("\n".join(all_trues))
    with open(f"{dataset_path}/falses_up_to_{i}.txt", "w") as f:
        f.write("\n".join(all_falses))


# Now make the "up to length" datasets for all the lengths and languages
def make_all_up_to_length_datasets(length):
    for language_generator in language_generators:
        for i in range(1, length + 1):
            dataset_path = f"{DATASET_PATH}/{language_generator.name}"
            make_up_to_length_dataset(dataset_path, i)


# Load in a dataset of up to length "length"
def load_adversarial_dataset(language_generator_name: str, length: int):
    dataset_path = f"{DATASET_PATH}/{language_generator_name}"
    with open(f"{dataset_path}/trues_up_to_{length}.txt", "r") as f:
        trues = f.read().splitlines()
    with open(f"{dataset_path}/falses_up_to_{length}.txt", "r") as f:
        falses = f.read().splitlines()

    # Make the dataset as a dict with "text" and "label" keys
    dataset = {"text": trues + falses, "label": [1] * len(trues) + [0] * len(falses)}
    # TODO: do we need to shuffle it too, or is that managed by the Trainer?

    return dataset


if __name__ == "__main__":
    make_all_up_to_length_datasets(5)
