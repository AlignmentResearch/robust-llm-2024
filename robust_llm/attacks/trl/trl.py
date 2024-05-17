import os
from typing import Any, Optional, Sequence, Tuple

import torch
import wandb
from datasets import Dataset
from tqdm import tqdm
from trl import PPOTrainer
from typing_extensions import override

from robust_llm import logger
from robust_llm.attacks.attack import Attack
from robust_llm.attacks.trl.utils import (
    LogitTextClassificationPipeline,
    check_for_not_finite,
    make_ppo_trainer,
    prepare_adversary,
    prepare_prompts,
)
from robust_llm.config.attack_configs import TRLAttackConfig
from robust_llm.models import WrappedModel
from robust_llm.rllm_datasets.modifiable_chunk_spec import ModifiableChunkSpec
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset

TRL_RESPONSE_STR = "<USER INPUT HERE>"


class TRLAttack(Attack):
    """Transformer Reinforcement Learning attack.

    Replaces all the modifiable text with random tokens
    from the tokenizer's vocabulary.
    """

    REQUIRES_TRAINING = True

    def __init__(
        self,
        attack_config: TRLAttackConfig,
        logging_name: str,
        victim: WrappedModel,
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
            logger.warning(
                "If you want to log trl training stats, "
                "you need to set a positive log_frequency. "
                "As is, no trl train stats will be logged."
            )

        # TODO (GH#374): Remove pipeline.
        self.victim_pipeline = LogitTextClassificationPipeline(
            model=victim,
            tokenizer=victim.tokenizer,
            # If we do not pass the device, the pipeline will move
            # self.victim to cpu as a side-effect
            device=victim.device,
            # We have to specify the framework explicitly because the pipeline
            # will not be able to infer it from a WrappedModel.
            framework="pt",
        )

        self.victim_batch_size = attack_config.victim_inference_batch_size
        self.adversary_batch_size = attack_config.batch_size

        self.reward_type = attack_config.reward_type

        self.seed = attack_config.seed
        self.device = victim.model.device

        self.model_name_to_save = attack_config.model_name_to_save
        self.model_save_path_prefix = attack_config.model_save_path_prefix

        assert victim.accelerator is not None
        self.adversary = prepare_adversary(
            attack_config, victim.model.config.num_labels, victim.accelerator
        )

        # NOTE: these values are taken from the TRL quickstart example
        # and might not be optimal for this setting
        # defaults are here
        #   https://huggingface.co/docs/transformers/v4.36.1/en/main_classes/text_generation#transformers.GenerationConfig
        # TRL quickstart example is here
        #   https://huggingface.co/docs/trl/v0.7.4/en/quickstart#minimal-example
        self.generation_kwargs = {
            "min_length": attack_config.min_length,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.adversary.tokenizer.eos_token_id,
            "max_new_tokens": attack_config.max_new_tokens,
        }
        self.ppo_trainer: Optional[PPOTrainer] = None

    @override
    def train(
        self,
        dataset: RLLMDataset,
    ) -> None:
        batch_size = self.adversary_batch_size
        assert batch_size <= len(dataset.ds)

        assert isinstance(self.attack_config, TRLAttackConfig)
        self.ppo_trainer = make_ppo_trainer(
            attack_config=self.attack_config,
            adversary_model=self.adversary.model,
            adversary_tokenizer=self.adversary.tokenizer,
            dataset=dataset.ds,
        )

        assert self.ppo_trainer is not None
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
                    modifiable_chunk_spec=dataset.modifiable_chunk_spec,
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

            logger.info(
                "Training TRL; epoch %s had average reward %s", epoch, average_reward
            )

        self._maybe_save_model_to_path_or_hf()

    def _get_reward(self, label: int, logits: torch.Tensor) -> torch.Tensor:
        reward_type = self.reward_type

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
        assert dataset.modifiable_chunk_spec.n_modifiable_chunks == 1

        attacked_texts, _, _ = self._get_attacked_texts(
            dataset=dataset.ds,
            modifiable_chunk_spec=dataset.modifiable_chunk_spec,
        )

        attacked_dataset = dataset.with_attacked_text(attacked_texts)
        return attacked_dataset, {}

    def _get_attacked_texts(
        self,
        dataset: Dataset,
        modifiable_chunk_spec: ModifiableChunkSpec,
    ) -> Tuple[Sequence[str], Sequence[torch.Tensor], Sequence[torch.Tensor]]:
        """The trl attack method itself.

        Args:
            dataset: The dataset to attack. Must have a "chunked_text" column.
            modifiable_chunk_spec: Specification which chunks of the original text can
            be modified, and how.

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
            modifiable_chunk_spec=modifiable_chunk_spec,
        )
        context_tensors = self.adversary.tokenizer(
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

        adversary_generated_responses_txt = self.adversary.tokenizer.batch_decode(
            adversary_generated_responses
        )

        attacked_texts = prepare_prompts(
            text_chunked=chunked_text,
            response_text=adversary_generated_responses_txt,
            modifiable_chunk_spec=modifiable_chunk_spec,
        )

        return (
            attacked_texts,
            context_tensor_list,
            adversary_generated_responses,
        )

    def _maybe_save_model_to_path_or_hf(self) -> None:

        if self.model_save_path_prefix is None:
            logger.warning("No model_save_path_prefix provided; not saving the model")
            return
        assert wandb.run is not None
        output_dir = os.path.join(
            self.model_save_path_prefix, "models", self.model_name_to_save
        )
        wandb.run.summary["saved_dir"] = output_dir  # type: ignore
        logger.info("Saving the trl model to %s", output_dir)

        # TODO(niki): enable saving on hf hub
        self.adversary.model.save_pretrained(output_dir)
        self.adversary.tokenizer.save_pretrained(output_dir)
