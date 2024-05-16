from transformers import AutoTokenizer

from robust_llm.config.configs import DatasetConfig
from robust_llm.models import GPTNeoXModel
from robust_llm.models.model_utils import InferenceType
from robust_llm.rllm_datasets.load_rllm_dataset import load_rllm_dataset
from robust_llm.training import _get_only_data_with_incorrect_predictions
from robust_llm.utils import FakeClassifierWithPositiveList


def test_get_only_data_with_incorrect_predictions():
    cfg = DatasetConfig(
        dataset_type="AlignmentResearch/PasswordMatch",
        n_train=10,
        n_val=10,
    )
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train = load_rllm_dataset(cfg, split="train").tokenize(tokenizer)

    # assume all are marked positive
    positives = tokenizer.batch_encode_plus(
        train.ds["text"], padding=True, return_tensors="pt"
    ).input_ids
    model = FakeClassifierWithPositiveList(tokenizer=tokenizer, positives=positives)
    # We ignore the type because we are using a FakeClassifierWithPositiveList
    victim = GPTNeoXModel(
        model,  # type: ignore
        tokenizer,
        accelerator=None,
        inference_type=InferenceType("classification"),
    )

    subset_indices = [
        i for i, d in enumerate(train.ds) if d["clf_label"] == 0  # type: ignore
    ]
    expected_filtered_dataset = train.get_subset(subset_indices)

    filtered_dataset = _get_only_data_with_incorrect_predictions(
        dataset=train,
        victim=victim,
        batch_size=2,
    )

    assert filtered_dataset.ds["text"] == expected_filtered_dataset.ds["text"]
    assert filtered_dataset.ds["clf_label"] == expected_filtered_dataset.ds["clf_label"]
