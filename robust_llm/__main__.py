from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from robust_llm.language_generators.tomita1 import Tomita1
from robust_llm.training import Training


def tokenize_dataset(dataset, tokenizer):
    # Padding seems necessary in order to avoid an error
    tokenized_data = tokenizer(dataset["text"], padding="max_length", truncation=True)
    return {"text": dataset["text"], "label": dataset["label"], **tokenized_data}


def get_overlap(smaller_set, larger_set):
    overlap = []
    for s in smaller_set["text"]:
        if s in larger_set["text"]:
            overlap.append(s)
    return overlap


def main():
    model_name = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    tomita1 = Tomita1(
        500, seed=41
    )  # <= 509 since we need to stay within the context of BERT
    # thought: maybe this doesn't generalize to longer lengths?

    train_size = 1000
    val_size = 200
    test_size = 200

    print("train_size", train_size)
    print("val_size", val_size)
    print("test_size", test_size)

    train_set, val_set, test_set = tomita1.generate_dataset(
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
    )

    # print("First ten train examples:")
    # for i in range(10):
    #     print(train_set["text"][i], train_set["label"][i])

    # How much of val set is in train set?
    train_val_overlap = get_overlap(smaller_set=val_set, larger_set=train_set)
    print("train val overlap size", len(train_val_overlap))
    print("train val overlap proportion", len(train_val_overlap) / len(val_set["text"]))
    print()

    # How much of test set is in train set?
    train_test_overlap = get_overlap(smaller_set=test_set, larger_set=train_set)
    print("train test overlap size", len(train_test_overlap))
    print("train test overlap proportion", len(train_test_overlap) / len(test_set["text"]))
    print()

    # How much of test set is in val set?
    val_test_overlap = get_overlap(smaller_set=test_set, larger_set=val_set)
    print("val test overlap size", len(val_test_overlap))
    print("val test overlap proportion", len(val_test_overlap) / len(test_set["text"]))
    print()

    # tokenized_train_dataset = Dataset.from_dict(tokenize_dataset(train_set, tokenizer))
    # tokenized_val_dataset = Dataset.from_dict(tokenize_dataset(val_set, tokenizer))
    # tokenized_test_dataset = Dataset.from_dict(tokenize_dataset(test_set, tokenizer))
    #
    # print("First ten tokenized train examples:")
    # for i in range(10):
    #     print(tokenized_train_dataset["text"][i], tokenized_train_dataset["label"][i])
    #
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     "bert-base-cased", num_labels=2
    # )
    #
    # training = Training(
    #     hparams={},
    #     train_dataset=tokenized_train_dataset,
    #     eval_dataset=tokenized_val_dataset,
    #     model=model,
    # )
    # training.run_trainer()


if __name__ == "__main__":
    main()
