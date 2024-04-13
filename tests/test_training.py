from datasets import Dataset
from transformers import AutoTokenizer

from robust_llm.training import _get_only_data_with_incorrect_predictions
from robust_llm.utils import FakeClassifierWithPositiveList


def test_get_only_data_with_incorrect_predictions():
    DATA = [
        # text, label, prediction
        ("text1", 1, 0),
        ("longer_text_2", 0, 0),
        ("ssfsdGGGG", 1, 1),
        ("text4", 0, 1),
        ("text5", 1, 1),
        ("text6", 0, 0),
        ("text7", 1, 1),
        ("blahblah", 1, 0),
    ]

    dataset = Dataset.from_dict(
        {
            "text": [d[0] for d in DATA],
            "label": [d[1] for d in DATA],
        }
    )
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    positives = tokenizer.batch_encode_plus(
        [d[0] for d in DATA if d[2]], padding=True, return_tensors="pt"
    ).input_ids
    model = FakeClassifierWithPositiveList(tokenizer=tokenizer, positives=positives)

    expected_filtered_dataset = Dataset.from_dict(
        {
            "text": [d[0] for d in DATA if d[1] != d[2]],
            "label": [d[1] for d in DATA if d[1] != d[2]],
        }
    )

    filtered_dataset = _get_only_data_with_incorrect_predictions(
        dataset=dataset,
        model=model,  # type: ignore
        tokenizer=tokenizer,
        batch_size=2,
    )

    assert filtered_dataset["text"] == expected_filtered_dataset["text"]
    assert filtered_dataset["label"] == expected_filtered_dataset["label"]
