from robust_llm.language_generators.tomita1 import Tomita1
from robust_llm.language_generators.tomita2 import Tomita2
from robust_llm.language_generators.tomita4 import Tomita4
from robust_llm.language_generators.tomita7 import Tomita7


language_generators = {Tomita1, Tomita2, Tomita4, Tomita7}

for language_generator in language_generators:
    t = language_generator(max_length=50)

    for i in range(1, 26):
        print("making dataset", t, i)
        t.make_complete_dataset(i)
