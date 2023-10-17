from simple_parsing import ArgumentParser


def add_parser_arguments(parser):
    parser.add_argument(
        "--adversarial_training",
        type=bool,
        default=False,
        help="Whether to use adversarial training or not.",
    )
    parser.add_argument(
        "--proportion",
        type=float,
        default=0.1,
        help="The proportion of the brute force dataset to use for training, when running a baseline.",
    )
    parser.add_argument(
        "--brute_force_attack",
        type=bool,
        default=False,
        help="Whether to exhaustively test all possible adversarial examples or not.",
    )
    parser.add_argument(
        "--brute_force_length",
        type=int,
        default=-1,
        help="Up to which length strings should be exhaustively tested.",
    )
    parser.add_argument(
        "--min_num_adversarial_examples_to_add",
        type=int,
        default=50,
        help="The minimum number of adversarial examples to add to the train set each attack round.",
    )
    parser.add_argument(
        "--max_num_search_for_adversarial_examples",
        type=int,
        default=8192,
        help="The maximum number of examples to search for adversarial examples in each attack round. Think 'compute budget'.",
    )
    parser.add_argument(
        "--adversarial_example_search_minibatch_size",
        type=int,
        default=64,
        help="The size of the minibatches to use when searching for adversarial examples.",
    )
    parser.add_argument(
        "--language_generator",
        choices=["tomita1", "tomita2", "tomita4", "tomita7"],
        default="tomita4",
        help="Choose the regular language to use (tomita1, tomita2, tomita4, tomita7). "
        "Defaults to tomita4 because 1 and 2 are rather simple.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=50,
        help="The maximum length of the strings to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="The seed to use for the random number generator used to make the dataset.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="The number of epochs to train for.",
    )
    parser.add_argument(
        "--num_adversarial_training_rounds",
        type=int,
        default=3,
        help="The number of adversarial training rounds to do.",
    )
    parser.add_argument(
        "--train_set_size",
        type=int,
        default=100,
        help="The size of the train set.",
    )
    parser.add_argument(
        "--val_set_size",
        type=int,
        default=100,
        help="The size of the validation set.",
    )
    parser.add_argument(
        "--skip_first_training_round",
        type=bool,
        default=False,
        help="Whether to skip the first training round or not.",
    )
    parser.add_argument(
        "--use_probabilistic_robustness_check",
        type=bool,
        default=False,
        help="If true, only checks robustness on a random subset of the brute force attack dataset.",
    )


def setup_argument_parser():
    parser = ArgumentParser()
    add_parser_arguments(parser)
    return parser
