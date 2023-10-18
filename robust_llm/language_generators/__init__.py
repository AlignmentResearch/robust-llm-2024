from robust_llm.language_generators.tomita1 import Tomita1
from robust_llm.language_generators.tomita2 import Tomita2
from robust_llm.language_generators.tomita4 import Tomita4
from robust_llm.language_generators.tomita7 import Tomita7
from robust_llm.language_generators.tomita_base import TomitaBase


def make_language_generator(language_name: str, max_length: int) -> TomitaBase:
    language_generator: TomitaBase
    language_name = language_name.lower()
    if language_name == "tomita1":
        language_generator = Tomita1(max_length=max_length)
    elif language_name == "tomita2":
        language_generator = Tomita2(max_length=max_length)
    elif language_name == "tomita4":
        language_generator = Tomita4(max_length=max_length)
    elif language_name == "tomita7":
        language_generator = Tomita7(max_length=max_length)
    else:
        raise ValueError(f"Unsupported language: {language_name}")

    return language_generator
