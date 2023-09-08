from simple_parsing import ArgumentParser


def add_parser_arguments(parser):
    parser.add_argument(
        "--adversarial_training",
        type=bool,
        default=False,
        help="Whether to use adversarial training or not.",
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
        "--random_sample_attack",
        type=bool,
        default=False,
        help="Whether to randomly sample adversarial examples or not.",
    )
    parser.add_argument(
        "--language_generator",
        choices=["Tomita1", "Tomita2", "Tomita4", "Tomita7"],
        default="Tomita4",
        help="Choose the regular language to use (Tomita1, Tomita2, Tomita4, Tomita7). "
        "Defaults to Tomita4 because 1 and 2 are rather simple.",
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
        "--test_set_size",
        type=int,
        default=100,
        help="The size of the test set.",
    )


def setup_argument_parser():
    parser = ArgumentParser()
    add_parser_arguments(parser)
    return parser
