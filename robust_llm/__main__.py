# TODO import stuff
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data_preperation import Data
from training import Training


def main():
    dataset = Data(dataset_name="imdb")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    train_dataset, test_dataset = dataset.prepare_tokenized_dataset(
        tokenizer=tokenizer, mini=True
    )

    # print(train_dataset.shape, test_dataset.shape)
    # print(train_dataset[100]['text'])
    # print(train_dataset[100]['label'])

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=5
    )
    # print(model)

    training = Training(
        hparams={},
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        model=model,
    )
    training.run_trainer()


if __name__ == "__main__":
    main()
