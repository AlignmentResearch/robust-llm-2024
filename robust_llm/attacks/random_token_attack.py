from typing import Optional

import torch
import transformers
from datasets import Dataset
from typing_extensions import override

from robust_llm.attacks.attack import Attack
from robust_llm.configs import AttackConfig
from robust_llm.dataset_management.dataset_management import ModifiableChunksSpec


class RandomTokenAttack(Attack):
    """Random token attack for non-Tomita datasets.

    Replaces all the modifiable text with random tokens
    from the tokenizer's vocabulary.
    """

    def __init__(
        self,
        attack_config: AttackConfig,
        modifiable_chunks_spec: ModifiableChunksSpec,
        dataset_type: str,
        tokenizer: transformers.PreTrainedTokenizerBase,
    ) -> None:
        """Constructor for RandomTokenAttack.

        Args:
            attack_config: config of the attack
            modifiable_chunks_spec: Specification for which chunks of the
                original text can be modified
            dataset_type: used dataset type
            tokenizer: tokenizer used with the model
        """
        super().__init__(attack_config, modifiable_chunks_spec)

        self.tokenizer = tokenizer
        self.torch_rng = torch.random.manual_seed(self.attack_config.seed)

        if dataset_type == "tomita":
            raise ValueError(
                "Random token attack is not supported for dataset type "
                f"{dataset_type}, exiting..."
            )

        assert True in modifiable_chunks_spec

    @override
    def get_attacked_dataset(
        self,
        dataset: Optional[Dataset] = None,
        max_n_outputs: Optional[int] = None,
    ) -> Dataset:

        assert dataset is not None and "text_chunked" in dataset.column_names

        text_chunked = dataset["text_chunked"]
        attacked_text_chunked = []

        min_tokens = self.attack_config.random_token_attack_attack_config.min_tokens
        max_tokens = self.attack_config.random_token_attack_attack_config.max_tokens

        # Replace all the modifiable text with random tokens
        # from the tokenizer's vocabulary.
        for line in text_chunked:
            new_line = []
            for text, is_modifiable in zip(line, self.modifiable_chunks_spec):
                if is_modifiable:
                    num_tokens = int(
                        torch.randint(
                            low=min_tokens,
                            high=max_tokens + 1,
                            size=(),
                            generator=self.torch_rng,
                        ).item()
                    )
                    random_tokens = torch.randint(
                        low=0,
                        high=self.tokenizer.vocab_size,  # type: ignore
                        size=(num_tokens,),
                        generator=self.torch_rng,
                    )

                    random_token_text = self.tokenizer.decode(random_tokens)
                    new_line.append(random_token_text)
                else:
                    new_line.append(text)
            attacked_text_chunked.append(new_line)

        # Also replace dataset text, and delete tokenized text
        attacked_text = ["".join(line) for line in attacked_text_chunked]

        new_dataset = Dataset.from_dict(
            {
                "text": attacked_text,
                "label": dataset["label"],
            }
        )

        return new_dataset
