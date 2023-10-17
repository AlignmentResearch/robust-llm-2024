from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import wandb
from robust_llm.experiments.adversarial_training import adversarial_training
from robust_llm.experiments.brute_force_baseline import brute_force_baseline
from robust_llm.language_generators import make_language_generator
from robust_llm.parsing import setup_argument_parser
from robust_llm.training import AdversarialTraining, Training
from robust_llm.utils import get_overlap, tokenize_dataset


def main():
    brute_force_baseline.main()


if __name__ == "__main__":
    main()
