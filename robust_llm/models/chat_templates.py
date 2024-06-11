from robust_llm.models.model_utils import PromptTemplate


def get_tinyllama_template(
    unmodifiable_prefix: str, modifiable_infix: str, unmodifiable_suffix: str
) -> PromptTemplate:
    before_attack = f"<|user|>\n{unmodifiable_prefix}{modifiable_infix}"
    after_attack = f"{unmodifiable_suffix}\n<|assistant|>"

    return PromptTemplate(
        before_attack=before_attack,
        after_attack=after_attack,
    )


def get_base_template(
    unmodifiable_prefix: str, modifiable_infix: str, unmodifiable_suffix: str
) -> PromptTemplate:
    before_attack = unmodifiable_prefix + modifiable_infix
    after_attack = unmodifiable_suffix

    return PromptTemplate(
        before_attack=before_attack,
        after_attack=after_attack,
    )


def get_llama_2_template(
    unmodifiable_prefix: str, modifiable_infix: str, unmodifiable_suffix: str
) -> PromptTemplate:
    before_attack = f"<s> [INST] {unmodifiable_prefix}{modifiable_infix}"
    after_attack = f"{unmodifiable_suffix} [/INST]"

    return PromptTemplate(
        before_attack=before_attack,
        after_attack=after_attack,
    )


def get_llama_3_template(
    unmodifiable_prefix: str, modifiable_infix: str, unmodifiable_suffix: str
) -> PromptTemplate:
    # To save some space we define shorter aliases.
    user_block = "<|start_header_id|>user<|end_header_id|>"
    asst_block = "<|start_header_id|>assistant<|end_header_id|>"
    up = unmodifiable_prefix
    mi = modifiable_infix

    before_attack = f"<|begin_of_text|>{user_block}\n\n{up}{mi}"
    after_attack = f"{unmodifiable_suffix}<|eot_id|>{asst_block}\n\n"

    return PromptTemplate(
        before_attack=before_attack,
        after_attack=after_attack,
    )


def get_qwen_template(
    unmodifiable_prefix: str, modifiable_infix: str, unmodifiable_suffix: str
) -> PromptTemplate:
    # To save some space we define shorter aliases.
    syst_block = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    up = unmodifiable_prefix
    mi = modifiable_infix

    before_attack = f"{syst_block}<|im_start|>user\n{up}{mi}"
    after_attack = f"{unmodifiable_suffix}<|im_end>\n<|im_start|>assistant\n"

    return PromptTemplate(
        before_attack=before_attack,
        after_attack=after_attack,
    )
