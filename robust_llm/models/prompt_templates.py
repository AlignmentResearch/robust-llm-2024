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
