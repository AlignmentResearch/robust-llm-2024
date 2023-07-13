from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from robust_llm.language_generators.tomita1 import Tomita1
from robust_llm.training import Training

def tokenize_dataset(dataset, tokenizer):
    # Padding seems necessary in order to avoid an error
    tokenized_data = tokenizer(dataset["text"], padding="max_length", truncation=True)
    return {"text": dataset["text"], "label": dataset["label"], **tokenized_data}


def main():
    model_name = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    tomita1 = Tomita1(5000, seed=41)  # Need to stay within the context of BERT
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

    print("First ten train examples:")
    for i in range(10):
        print(train_set["text"][i], train_set["label"][i])

    # Check the size of the overlap between train set and val set and test set
    train_val_overlap = (
        len(set(train_set["text"]).intersection(set(val_set["text"]))) / val_size
    )
    train_test_overlap = (
        len(set(train_set["text"]).intersection(set(test_set["text"]))) / test_size
    )
    val_test_overlap = (
        len(set(val_set["text"]).intersection(set(test_set["text"]))) / test_size
    )

    print("train val overlap", train_val_overlap)
    print("train test overlap", train_test_overlap)
    print("val test overlap", val_test_overlap)

    tokenized_train_dataset = Dataset.from_dict(tokenize_dataset(train_set, tokenizer))
    tokenized_val_dataset = Dataset.from_dict(tokenize_dataset(val_set, tokenizer))
    tokenized_test_dataset = Dataset.from_dict(tokenize_dataset(test_set, tokenizer))

    print("First ten tokenized train examples:")
    for i in range(10):
        print(tokenized_train_dataset["text"][i], tokenized_train_dataset["label"][i])

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=2
    )

    training = Training(
        hparams={},
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        model=model,
    )
    training.run_trainer()


if __name__ == "__main__":
    main()
