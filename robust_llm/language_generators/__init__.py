from robust_llm.language_generators.tomita1 import Tomita1, TomitaBase
from robust_llm.language_generators.tomita2 import Tomita2
from robust_llm.language_generators.tomita4 import Tomita4
from robust_llm.language_generators.tomita7 import Tomita7


def make_language_generator(language_name: str, max_length: int) -> TomitaBase:
    language_generator: TomitaBase
    if language_name == "Tomita1":
        language_generator = Tomita1(max_length=max_length)
    elif language_name == "Tomita2":
        language_generator = Tomita2(max_length=max_length)
    elif language_name == "Tomita4":
        language_generator = Tomita4(max_length=max_length)
    elif language_name == "Tomita7":
        language_generator = Tomita7(max_length=max_length)
    else:
        raise ValueError(f"Unsupported language: {language_name}")

    return language_generator
