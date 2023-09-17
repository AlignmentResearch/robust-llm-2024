from robust_llm.language_generators.tomita1 import Tomita1
from robust_llm.language_generators.tomita2 import Tomita2
from robust_llm.language_generators.tomita4 import Tomita4
from robust_llm.language_generators.tomita7 import Tomita7


def make_language_generator_from_args(args):
    if args.language_generator == "Tomita1":
        language_generator = Tomita1(max_length=args.max_length)
    elif args.language_generator == "Tomita2":
        language_generator = Tomita2(max_length=args.max_length)
    elif args.language_generator == "Tomita4":
        language_generator = Tomita4(max_length=args.max_length)
    elif args.language_generator == "Tomita7":
        language_generator = Tomita7(max_length=args.max_length)
    else:
        raise ValueError(f"Unknown language generator: {args.language_generator}")

    return language_generator
