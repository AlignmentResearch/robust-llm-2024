from robust_llm.dataset_management.tomita.tomita import Tomita
from robust_llm.dataset_management.tomita.tomita1 import Tomita1
from robust_llm.dataset_management.tomita.tomita2 import Tomita2
from robust_llm.dataset_management.tomita.tomita4 import Tomita4
from robust_llm.dataset_management.tomita.tomita7 import Tomita7


def make_language_generator(language_name: str, max_length: int) -> Tomita:
    language_generator: Tomita
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
