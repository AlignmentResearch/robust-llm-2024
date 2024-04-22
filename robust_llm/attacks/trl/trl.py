import os
from typing import Any, Optional, Sequence, Tuple

import torch
import wandb
from datasets import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
from trl import PPOConfig, PPOTrainer
from typing_extensions import override

from robust_llm.attacks.attack import Attack
from robust_llm.attacks.trl.utils import (
    LogitTextClassificationPipeline,
    check_for_not_finite,
    make_ppo_trainer,
    prepare_adversary_model_and_tokenizer,
    prepare_prompts,
)
from robust_llm.configs import AttackConfig
from robust_llm.rllm_datasets.dataset_utils import ModifiableChunksSpec
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset
from robust_llm.utils import LanguageModel

TRL_RESPONSE_STR = "<USER INPUT HERE>"


class TRLAttack(Attack):
    """Transformer Reinforcement Learning attack.

    Replaces all the modifiable text with random tokens
    from the tokenizer's vocabulary.
    """

    REQUIRES_TRAINING = True

    def __init__(
        self,
        attack_config: AttackConfig,
        logging_name: str,
        victim_model: LanguageModel,
        victim_tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        """Constructor for TRLAttack.

        Args:
            attack_config: config of the attack
            logging_name: name of the attack; used for logging
            victim_model: the model to be attacked
            victim_tokenizer: tokenizer used by the victim model
        """

        super().__init__(
            attack_config=attack_config,
            logging_name=logging_name,
        )

        # Check the logging frequency
        if self.attack_config.log_frequency is None:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(
                "If you want to log trl training stats, "
                "you need to set a positive log_frequency. "
                "As is, no trl train stats will be logged."
            )
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        self.victim_tokenizer = victim_tokenizer
        self.victim_pipeline = LogitTextClassificationPipeline(
            model=victim_model,
            tokenizer=victim_tokenizer,
            # If we do not pass the device, the pipeline will move
            # self.victim_model to cpu as a side-effect
            device=victim_model.device,
        )

        self.victim_batch_size = attack_config.victim_inference_batch_size

        self.seed = attack_config.seed
        self.device = victim_model.device

        self.model_name_to_save = attack_config.trl_attack_config.model_name_to_save

        (
            self.adversary_model,
            self.adversary_tokenizer,
        ) = prepare_adversary_model_and_tokenizer(
            self.attack_config,
            num_classes=victim_model.config.num_labels,
            device=self.device,
        )

        # NOTE: these values are taken from the TRL quickstart example
        # and might not be optimal for this setting
        # defaults are here
        #   https://huggingface.co/docs/transformers/v4.36.1/en/main_classes/text_generation#transformers.GenerationConfig
        # TRL quickstart example is here
        #   https://huggingface.co/docs/trl/v0.7.4/en/quickstart#minimal-example
        self.generation_kwargs = {
            "min_length": self.attack_config.trl_attack_config.min_length,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.adversary_tokenizer.eos_token_id,
            "max_new_tokens": self.attack_config.trl_attack_config.max_new_tokens,
        }
        self.ppo_trainer: Optional[PPOTrainer] = None

    @override
    def train(
        self,
        dataset: RLLMDataset,
    ) -> None:
        batch_size = self.attack_config.trl_attack_config.batch_size
        assert batch_size <= len(dataset.ds)

        self.ppo_trainer = make_ppo_trainer(
            attack_config=self.attack_config,
            adversary_model=self.adversary_model,
            adversary_tokenizer=self.adversary_tokenizer,
            dataset=dataset.ds,
        )

        assert self.ppo_trainer is not None
        assert isinstance(self.ppo_trainer.config, PPOConfig)
        ppo_epochs: int = self.ppo_trainer.config.ppo_epochs
        for epoch in tqdm(range(ppo_epochs), "epoch: "):
            epoch_rewards = []
            for batch in tqdm(self.ppo_trainer.dataloader):
                # Get the attacks from the adversary
                (
                    attacked_texts,
                    context_tensor_list,
                    responses,
                ) = self._get_attacked_texts(
                    batch,
                    modifiable_chunks_spec=dataset.modifiable_chunks_spec,
                )

                # Get victim responses
                victim_outputs = self.victim_pipeline(
                    attacked_texts, batch_size=self.victim_batch_size
                )
                logits = [logit.squeeze() for logit in victim_outputs]  # type: ignore

                # Adjust the ground truth labels to be consistent
                # with the new passwords generated by the adversary
                labels = batch["clf_label"]
                labels = dataset.maybe_recompute_labels(attacked_texts, labels)

                rewards = []
                for label, example_logits in zip(labels, logits):
                    example_logits = example_logits.clone().detach()
                    reward = self._get_reward(label, example_logits)
                    rewards.append(reward)

                train_stats = self.ppo_trainer.step(
                    queries=context_tensor_list,  # type: ignore
                    responses=responses,  # type: ignore
                    scores=rewards,
                )
                epoch_rewards.extend(rewards)

                # Check for not-finite values in the training stats
                check_for_not_finite(train_stats)

                # Update step and datapoints trained
                assert len(rewards) == len(context_tensor_list) == len(responses)

                # Log the ppo stats and update the logging counters
                self._maybe_log_trl(train_stats, rewards)

            average_reward = torch.Tensor(epoch_rewards).squeeze().mean()
            print(f"Training TRL; epoch {epoch} had average reward {average_reward}")

        self._maybe_save_model_to_path_or_hf()

    def _get_reward(self, label: int, logits: torch.Tensor) -> torch.Tensor:
        reward_type = self.attack_config.trl_attack_config.reward_type

        if reward_type == "minus_correct_logit_plus_incorrect_logits":
            return logits.sum() - 2 * logits[label]

        if reward_type == "minus_correct_logprob":
            logprobs = torch.nn.functional.log_softmax(logits, dim=0)
            return -logprobs[label]

        if reward_type == "minus_correct_prob":
            probs = torch.nn.functional.softmax(logits, dim=0)
            return -probs[label]

        raise ValueError(f"Reward type {reward_type} not recognized")

    def _maybe_log_trl(
        self, train_stats: dict[str, Any], rewards: Sequence[torch.Tensor]
    ):
        self.logging_counter.increment(
            step_count_to_add=1,
            datapoint_count_to_add=len(rewards),
            commit=False,
        )

        if self.attack_config.log_frequency is not None:
            if self.logging_counter.step_count % self.attack_config.log_frequency == 0:
                prepended_train_stats = {
                    f"{self.logging_name}/{key}": value
                    for key, value in train_stats.items()
                }

                wandb.log(prepended_train_stats, commit=True)

    @override
    def get_attacked_dataset(
        self,
        dataset: RLLMDataset,
    ) -> tuple[RLLMDataset, dict[str, Any]]:

        # At present, the trl attack is set up to only work
        # with one modifiable chunk
        assert sum(dataset.modifiable_chunks_spec) == 1  # exactly one True

        attacked_texts, _, _ = self._get_attacked_texts(
            dataset=dataset.ds,
            modifiable_chunks_spec=dataset.modifiable_chunks_spec,
        )

        attacked_dataset = dataset.with_attacked_text(attacked_texts)
        return attacked_dataset, {}

    def _get_attacked_texts(
        self,
        dataset: Dataset,
        modifiable_chunks_spec: ModifiableChunksSpec,
    ) -> Tuple[Sequence[str], Sequence[torch.Tensor], Sequence[torch.Tensor]]:
        """The trl attack method itself.

        Args:
            dataset: The dataset to attack. Must have a "chunked_text" column.
            modifiable_chunks_spec: Specification which chunks of the original text can
            be modified.

        Returns:
            attacked_texts: A sequence of the full attacked texts, ready to be
                passed to the victim model for classification. Made by combining
                the context with the adversary-generated responses.
            context_tensor_list: A sequence of tensors, each representing
                the context that was passed to the adversary model.
            adversary_generated_responses: A sequence of tensors, each representing
                the response generated by the adversary model
        """

        if self.ppo_trainer is None:
            raise ValueError(
                "You must train the TRLAttack before using it to generate attacks"
            )

        chunked_text = dataset["chunked_text"]
        contexts = prepare_prompts(
            text_chunked=chunked_text,
            response_text=TRL_RESPONSE_STR,
            modifiable_chunks_spec=modifiable_chunks_spec,
            append_to_modifiable_chunk=self.attack_config.append_to_modifiable_chunk,  # noqa: E501
        )
        context_tensors = self.adversary_tokenizer(
            contexts, padding="max_length", truncation=True, return_tensors="pt"  # type: ignore # noqa: E501
        )
        context_tensor_list = [
            context_tensor for context_tensor in context_tensors["input_ids"]  # type: ignore # noqa: E501
        ]

        with torch.no_grad():
            adversary_generated_responses = self.ppo_trainer.generate(
                context_tensor_list,
                return_prompt=False,
                **self.generation_kwargs,
            )

        # Check that the ppo trainer's output is a list of lists of tensors
        assert isinstance(adversary_generated_responses, list)
        assert all(
            isinstance(response, torch.Tensor)
            for response in adversary_generated_responses
        )

        adversary_generated_responses_txt = self.adversary_tokenizer.batch_decode(
            adversary_generated_responses
        )

        attacked_texts = prepare_prompts(
            text_chunked=chunked_text,
            response_text=adversary_generated_responses_txt,
            modifiable_chunks_spec=modifiable_chunks_spec,
            append_to_modifiable_chunk=self.attack_config.append_to_modifiable_chunk,  # noqa: E501
        )

        return (
            attacked_texts,
            context_tensor_list,
            adversary_generated_responses,
        )

    def _maybe_save_model_to_path_or_hf(self) -> None:

        model_save_path_prefix = (
            self.attack_config.trl_attack_config.model_save_path_prefix
        )

        assert model_save_path_prefix is not None
        assert wandb.run is not None
        output_dir = os.path.join(
            model_save_path_prefix, "models", self.model_name_to_save
        )
        wandb.run.summary["saved_dir"] = output_dir  # type: ignore
        print(f"Saving the trl model to {output_dir}")

        self.adversary_model.save_pretrained(output_dir)
        self.adversary_tokenizer.save_pretrained(output_dir)
