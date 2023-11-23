import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import sys
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GPTNeoXForSequenceClassification,
    GPTNeoXTokenizerFast,
)
from transformers.modeling_utils import PreTrainedModel
import yaml

import wandb

from robust_llm.configs import OverallConfig
from robust_llm.dataset_management.dataset_management import (
    generateRobustLLMDatasets,
)
from robust_llm.dataset_management.tomita import make_language_generator
from robust_llm.experiment_scripts import scaling_experiments
from robust_llm.training import AdversarialTraining, Training
from robust_llm.utils import get_overlap


cs = ConfigStore.instance()
cs.store(name="base_config", node=OverallConfig)


@hydra.main(version_base=None, config_path="hydra_conf", config_name="default_config")
def main(args: OverallConfig) -> None:
    experiment = args.experiment
    if experiment.scaling_experiments:
        print("Running a scaling experiment...")
        hydra_config_name = HydraConfig.get().overrides.task[0].split("=")[1]
        scaling_experiments.run_experiment(
            experiment_yaml=hydra_config_name,
            experiment_name=experiment.experiment_name,
        )
        return

    print("Configuration arguments:\n")
    print(OmegaConf.to_yaml(experiment))
    print()

    if experiment.environment.dataset_type.lower() == "tensor_trust":
        language_generator = None
        dataset_type = "tensor_trust"
    elif experiment.environment.dataset_type.lower() == "tomita":
        dataset_type = "tomita"
        language_generator = make_language_generator(
            experiment.environment.language_generator, experiment.environment.max_length
        )
    else:
        raise ValueError(f"Unknown dataset type {experiment.environment.dataset_type}")

    # Choose a model and a tokenizer
    model_name = args.experiment.environment.model_name

    if "bert" in model_name:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    elif "pythia" in model_name:
        checkpoint_step_number: int = experiment.training.checkpoint
        checkpoint_string: str = f"step{checkpoint_step_number}"
        pythia_version = model_name.split("/")[-1]
        untyped_model = GPTNeoXForSequenceClassification.from_pretrained(
            model_name,
            revision=checkpoint_string,
            cache_dir=f"./{pythia_version}/{checkpoint_string}",
            use_cache=False,  # otherwise returns last key/values attentions
            num_labels=2,
        )
        assert isinstance(untyped_model, PreTrainedModel)
        model = untyped_model
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=checkpoint_string,
            cache_dir=f"./{pythia_version}/{checkpoint_string}",
            use_fast=True,
            model_max_length=512,  # TODO: check this number
        )
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    else:
        raise ValueError(f"Unknown model name {model_name}")

    robust_llm_datasets = generateRobustLLMDatasets(
        dataset_type, language_generator, tokenizer, experiment.training
    )

    # NOTE: the "validation" dataset is one of what will be
    # several datasets that we perform model evaluation on,
    # hence "eval_dataset" is a dict[str, Dataset], not a Dataset.
    base_training_args = {
        "hparams": {},
        "experiment_name": experiment.experiment_name,
        "run_name": experiment.run_name,
        "job_type": experiment.job_type,
        "train_dataset": robust_llm_datasets.tokenized_train_dataset,
        "eval_dataset": {
            "validation": robust_llm_datasets.tokenized_validation_dataset
        },
        "model": model,
        "train_epochs": experiment.training.num_train_epochs,
    }

    # Set up the training environment
    training: Training
    if experiment.training.iterative.iterative_training:
        training = AdversarialTraining(
            **base_training_args,
            num_iterative_training_rounds=experiment.training.iterative.num_iterative_training_rounds,
            tokenizer=tokenizer,
            dataset_type=dataset_type,
            language_generator=language_generator,
            brute_force_attack=experiment.training.iterative.brute_force_attack,
            brute_force_length=experiment.training.iterative.brute_force_length,
            min_num_new_examples_to_add=experiment.training.iterative.min_num_new_examples_to_add,
            max_num_search_for_adversarial_examples=experiment.training.iterative.max_num_search_for_adversarial_examples,
            adversarial_example_search_minibatch_size=experiment.training.iterative.adversarial_example_search_minibatch_size,
            skip_first_training_round=experiment.training.iterative.skip_first_training_round,
            use_probabilistic_robustness_check=experiment.training.iterative.use_probabilistic_robustness_check,
            non_adversarial_baseline=experiment.training.iterative.non_adversarial_baseline,
        )
    else:
        training = Training(
            **base_training_args,
        )

    # Log the training arguments to wandb
    if not wandb.run:
        raise ValueError("wandb should have been initialized by now, exiting...")
    yaml_string = yaml.load(OmegaConf.to_yaml(experiment), Loader=yaml.FullLoader)
    wandb.run.summary[f"experiment_yaml"] = yaml_string

    # Log the train-val overlap to wandb
    if (
        experiment.training.train_set_size > 0
        and experiment.training.validation_set_size > 0
    ):
        if not wandb.run:
            raise ValueError("wandb should have been initialized by now, exiting...")
        train_val_overlap = get_overlap(
            smaller_dataset=robust_llm_datasets.validation_dataset,
            larger_dataset=robust_llm_datasets.train_dataset,
        )
        wandb.run.summary["train_val_overlap_size"] = len(train_val_overlap)
        wandb.run.summary["train_val_overlap_over_train_set_size"] = len(
            train_val_overlap
        ) / len(robust_llm_datasets.train_dataset["text"])
        wandb.run.summary["train_val_overlap_over_val_set_size"] = len(
            train_val_overlap
        ) / len(robust_llm_datasets.validation_dataset["text"])

    # Perform the training
    training.run_trainer()


if __name__ == "__main__":
    main()
