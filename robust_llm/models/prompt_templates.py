from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    """This is a general class for prompt templates,
    that should encompass both chat models and non-chat
    models

    The basic idea is that there is some part before user input, and
    some part after user input but before model input, and that should
    be all off the content in the prompt.

    For example, for a simple chat format:
        before_attack="User: Hi, I'm an user! "
        after_attack="\nAssistant:"
    """

    before_attack: str = ""
    after_attack: str = ""

    def build_prompt(self, *, attack_text: str = "", target: str = "") -> str:
        prompt = self.before_attack + attack_text + self.after_attack + target
        return prompt


@dataclass(frozen=True)
class PromptTemplateBuilder:
    """A template for generating chat prompts.

    Args:
        prompt_prefix (str): The prefix for the entire prompt.
        system_prefix (str): The prefix for the system prompt.
        system_suffix (str): The suffix for the system prompt.
        user_prefix (str): The prefix for the user input.
        user_suffix (str): The suffix for the user input, including the prefix
            for the assistant response.
        default_system_prompt (str | None, optional): The default system prompt.
            Defaults to None.

    Methods:
        get_prompt_template: Constructs a prompt template based on the provided
            prompt chunk strings.

    """

    prompt_prefix: str
    system_prefix: str
    system_suffix: str
    user_prefix: str
    user_suffix: str
    default_system_prompt: str | None = None

    def get_prompt_template(
        self,
        unmodifiable_prefix: str,
        modifiable_infix: str,
        unmodifiable_suffix: str,
        system_prompt: str | None = None,
    ):
        before_attack = self.prompt_prefix
        if system_prompt is not None:
            before_attack += f"{self.system_prefix}{system_prompt}{self.system_suffix}"
        elif self.default_system_prompt is not None:
            before_attack += (
                f"{self.system_prefix}{self.default_system_prompt}{self.system_suffix}"
            )
        before_attack += f"{self.user_prefix}{unmodifiable_prefix}{modifiable_infix}"
        after_attack = f"{unmodifiable_suffix}{self.user_suffix}"
        return PromptTemplate(
            before_attack=before_attack,
            after_attack=after_attack,
        )


def get_tinyllama_template(
    unmodifiable_prefix: str,
    modifiable_infix: str,
    unmodifiable_suffix: str,
    system_prompt: str | None = None,
) -> PromptTemplate:
    template = PromptTemplateBuilder(
        prompt_prefix="",
        system_prefix="<|system|>\n",
        system_suffix="</s>\n",
        user_prefix="<|user|>\n",
        user_suffix="</s>\n<|assistant|>\n",
    )
    return template.get_prompt_template(
        unmodifiable_prefix=unmodifiable_prefix,
        modifiable_infix=modifiable_infix,
        unmodifiable_suffix=unmodifiable_suffix,
        system_prompt=system_prompt,
    )


def get_base_template(
    unmodifiable_prefix: str,
    modifiable_infix: str,
    unmodifiable_suffix: str,
) -> PromptTemplate:
    template = PromptTemplateBuilder(
        prompt_prefix="",
        system_prefix="",
        system_suffix="",
        user_prefix="",
        user_suffix="",
    )
    return template.get_prompt_template(
        unmodifiable_prefix=unmodifiable_prefix,
        modifiable_infix=modifiable_infix,
        unmodifiable_suffix=unmodifiable_suffix,
    )


def get_llama_2_template(
    unmodifiable_prefix: str,
    modifiable_infix: str,
    unmodifiable_suffix: str,
    system_prompt: str | None = None,
) -> PromptTemplate:
    template = PromptTemplateBuilder(
        prompt_prefix="<s>[INST] ",
        system_prefix="<<SYS>>\n",
        system_suffix="\n<</SYS>>\n\n",
        user_prefix="",
        user_suffix=" [/INST] ",
    )
    return template.get_prompt_template(
        unmodifiable_prefix=unmodifiable_prefix,
        modifiable_infix=modifiable_infix,
        unmodifiable_suffix=unmodifiable_suffix,
        system_prompt=system_prompt,
    )


def get_llama_3_template(
    unmodifiable_prefix: str,
    modifiable_infix: str,
    unmodifiable_suffix: str,
    system_prompt: str | None = None,
) -> PromptTemplate:
    template = PromptTemplateBuilder(
        prompt_prefix="<|begin_of_text|>",
        system_prefix="<|start_header_id|>system<|end_header_id|>\n\n",
        system_suffix="<|eot_id|>",
        user_prefix="<|start_header_id|>user<|end_header_id|>\n\n",
        user_suffix="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    )
    return template.get_prompt_template(
        unmodifiable_prefix=unmodifiable_prefix,
        modifiable_infix=modifiable_infix,
        unmodifiable_suffix=unmodifiable_suffix,
        system_prompt=system_prompt,
    )


def get_qwen_template(
    unmodifiable_prefix: str,
    modifiable_infix: str,
    unmodifiable_suffix: str,
    system_prompt: str | None = None,
) -> PromptTemplate:
    template = PromptTemplateBuilder(
        prompt_prefix="",
        system_prefix="<|im_start|>system\n",
        system_suffix="<|im_end|>\n",
        user_prefix="<|im_start|>user\n",
        user_suffix="<|im_end|>\n<|im_start|>assistant\n",
        default_system_prompt="You are a helpful assistant.",
    )
    return template.get_prompt_template(
        unmodifiable_prefix=unmodifiable_prefix,
        modifiable_infix=modifiable_infix,
        unmodifiable_suffix=unmodifiable_suffix,
        system_prompt=system_prompt,
    )


def get_gemma_template(
    unmodifiable_prefix: str,
    modifiable_infix: str,
    unmodifiable_suffix: str,
) -> PromptTemplate:
    template = PromptTemplateBuilder(
        prompt_prefix="<bos>",
        system_prefix="",
        system_suffix="",
        user_prefix="<start_of_turn>user\n",
        user_suffix="<end_of_turn>\n<start_of_turn>model\n",
    )

    return template.get_prompt_template(
        unmodifiable_prefix=unmodifiable_prefix,
        modifiable_infix=modifiable_infix,
        unmodifiable_suffix=unmodifiable_suffix,
    )
