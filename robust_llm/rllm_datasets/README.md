# Robust-LLM Datasets
- This directory is intended to manage all creation and loading of datasets used in this project.
- Datasets are specified with a `DatasetConfig` and loaded with `load_rllm_dataset`.
- The [FAR AI huggingface organization](https://huggingface.co/AlignmentResearch) contains the authoritative versions of datasets.
    - See the [AlignmentResearch/IMDB repo](https://huggingface.co/datasets/AlignmentResearch/IMDB/tree/main) for an example.
    - Click the branch dropdown (labelled `main`) to see all versions.

## Versioning
- HF Datasets are versioned with https://semver.org/
    - Major versions correspond to breaking changes
        - E.g. we rename a column or save the data differently
        - We might change this when implementing generative tasks
        - Changes to the major version are tied to changes to the `DatasetUploadHandler` class: If that class changes in a way that affects the dataset format, the major version should be increased.
    - Minor versions correspond to new features
        - E.g. we add a column or add a new configuration
        - We might change this when adding columns used by adversarial attacks
    - Patch versions correspond to fixing bugs
        - E.g. we notice a label is wrong or we shuffled the data wrong
- When loading, `revision` specifies a version
    - E.g. `"1.0.3"`
    - Default is `"main"`, which should point to most up-to-date version of the dataset
- When loading, `config_name` specifies a configuration of the dataset
    - E.g. `"pos"` for only positive-class example
    - Default is `"default"`, which should point to the standard, full version of the dataset

## Generating datasets
- Dataset generation scripts can be found in `robust_llm/rllm_datasets/generation_scripts`.
- To add a new dataset, suggested workflows is:
    - If the dataset comes from huggingface and has two classes:
        - Copy `generate_imdb.py` and modify it with the relevant information.
    - If the dataset is generated from scratch:
        - Copy `generate_word_length.py` and modify it with the new dataset generation logic.
- To make a new version of an existing dataset, suggested workflow is:
    - Modify dataset generation inside `main` in the relevant script.
    - Increase the `MINOR_VERSION` and/or `PATCH_VERSION`.
        - The generation script will probably contain the most recently-used `MINOR_VERSION` and `PATCH_VERSION`, but check [huggingface](https://huggingface.co/AlignmentResearch) for a definitive list of existing versions.
