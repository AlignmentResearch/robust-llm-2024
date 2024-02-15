from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
from trl import PPOTrainer
from typing_extensions import override

from robust_llm.attacks.attack import Attack
from robust_llm.attacks.trl.utils import (
    LogitTextClassificationPipeline,
    make_ppo_trainer,
    prepare_adversary_model_and_tokenizer,
    prepare_prompts,
)
from robust_llm.configs import AttackConfig
from robust_llm.dataset_management.dataset_management import ModifiableChunksSpec
from robust_llm.utils import LanguageModel

TRL_RESPONSE_STR = "<USER INPUT HERE>"


class TRLAttack(Attack):
    """Transformer Reinforcement Learning attack.

    Works on non-Tomita datasets.
    Replaces all the modifiable text with random tokens
    from the tokenizer's vocabulary.
    """

    REQUIRES_INPUT_DATASET = True
    REQUIRES_TRAINING = True

    def __init__(
        self,
        attack_config: AttackConfig,
        modifiable_chunks_spec: ModifiableChunksSpec,
        dataset_type: str,
        victim_model: LanguageModel,
        victim_tokenizer: PreTrainedTokenizerBase,
        ground_truth_label_fn: Optional[Callable[[str], int]] = None,
    ) -> None:
        """Constructor for TRLAttack.

        Args:
            attack_config: config of the attack
            modifiable_chunks_spec: Specification for which chunks of the
                original text can be modified
            dataset_type: used dataset type
            victim_model: the model to be attacked
            victim_tokenizer: tokenizer used by the victim model
        """

        super().__init__(attack_config, modifiable_chunks_spec)

        # At present, the trl attack is set up to only work
        # with one modifiable chunk
        assert True in modifiable_chunks_spec
        assert sum(modifiable_chunks_spec) == 1  # exactly one True

        self.dataset_type = dataset_type
        self.victim_tokenizer = victim_tokenizer
        self.ground_truth_label_fn = ground_truth_label_fn
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

        if dataset_type == "tomita":
            raise ValueError(
                "TRL attack is not supported for dataset type "
                f"{dataset_type}, exiting..."
            )

        (
            self.adversary_model,
            self.adversary_tokenizer,
        ) = prepare_adversary_model_and_tokenizer(
            self.attack_config, device=self.device
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

    def train(
        self,
        dataset: Dataset,
    ) -> None:
        batch_size = self.attack_config.trl_attack_config.batch_size
        assert batch_size <= len(dataset["text"])

        self.ppo_trainer = make_ppo_trainer(
            attack_config=self.attack_config,
            adversary_model=self.adversary_model,
            adversary_tokenizer=self.adversary_tokenizer,
            dataset=dataset,
        )

        assert self.ppo_trainer is not None
        for epoch in tqdm(range(self.ppo_trainer.config.ppo_epochs), "epoch: "):
            epoch_rewards = []
            for batch in tqdm(self.ppo_trainer.dataloader):
                # Get the attacks from the adversary
                (
                    attacked_texts,
                    context_tensor_list,
                    responses,
                ) = self._get_attacked_texts(text_chunked=batch["text_chunked"])

                # Get victim responses
                victim_outputs = self.victim_pipeline(
                    attacked_texts, batch_size=self.victim_batch_size
                )
                logits = [logit.squeeze() for logit in victim_outputs]  # type: ignore

                # Adjust the ground truth labels to be consistent
                # with the new passwords generated by the adversary
                labels = batch["label"]
                if self.ground_truth_label_fn is not None:
                    labels = [
                        self.ground_truth_label_fn(attacked_text)
                        for attacked_text in attacked_texts  # type: ignore
                    ]

                rewards = []
                for label, logit_pair in zip(labels, logits):
                    detached_logit_pair = logit_pair.clone().detach()
                    if label == 0:
                        reward = detached_logit_pair[1] - detached_logit_pair[0]
                    else:
                        reward = detached_logit_pair[0] - detached_logit_pair[1]
                    rewards.append(reward)

                self.ppo_trainer.step(
                    queries=context_tensor_list,  # type: ignore
                    responses=responses,  # type: ignore
                    scores=rewards,  # type: ignore
                )

                epoch_rewards.extend(rewards)

                # TODO(niki): add logging here

            average_reward = torch.Tensor(epoch_rewards).squeeze().mean()
            print(f"Training TRL; epoch {epoch} had average reward {average_reward}")

    @override
    def get_attacked_dataset(
        self, dataset: Optional[Dataset], max_n_outputs: Optional[int] = None
    ) -> Tuple[Dataset, Dict[str, Any]]:
        if max_n_outputs is not None:
            raise ValueError("For now, max_n_outputs is not supported by TRLAttack")

        if dataset is None:
            raise ValueError("For now, dataset cannot be None for TRLAttack")

        if "text_chunked" not in dataset.column_names:
            raise ValueError(
                "Dataset must contain a 'text_chunked' column for TRLAttack"
            )

        attacked_texts, _, _ = self._get_attacked_texts(
            text_chunked=dataset["text_chunked"]
        )

        return (
            Dataset.from_dict(
                {
                    "text": attacked_texts,
                    "original_text": dataset["text"],
                    "label": dataset["label"],
                }
            ),
            {},
        )

    def _get_attacked_texts(
        self,
        text_chunked: Sequence[Sequence[str]],
    ) -> Tuple[Sequence[str], Sequence[torch.Tensor], Sequence[torch.Tensor]]:
        """The trl attack method itself.

        Args:
            text_chunked: Sequence of chunked texts. Each chunked text is
                a sequence of strings, some of which are modifiable,
                some of which are not, as determined by self.modifiable_chunks_spec.

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

        contexts = prepare_prompts(
            text_chunked=text_chunked,
            response_text=TRL_RESPONSE_STR,
            modifiable_chunks_spec=self.modifiable_chunks_spec,
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
            text_chunked=text_chunked,
            response_text=adversary_generated_responses_txt,
            modifiable_chunks_spec=self.modifiable_chunks_spec,
        )

        return (
            attacked_texts,
            context_tensor_list,
            adversary_generated_responses,
        )
