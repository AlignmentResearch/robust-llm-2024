# robust-llm

## Simple installation

If you just want to run the code and nothing else, you can do the following:

1. clone the repository
2. cd into it
3. create a new Python 3.10 virtual environment called `venv`
4. activate the virtual environment
5. install the `robust-llm` project

```
git clone https://github.com/AlignmentResearch/robust-llm.git
cd robust-llm
python -m venv venv
source venv/bin/activate
pip install .
```

Note that this project has not been tested with different versions of Python.

## Development installation

If you want to install `robust-llm` with developer dependencies (which gives you development tools like linting and tests), do the following:

1. Follow steps 1-4 from the [Simple installation](#simple-installation).

2. Add [pre-commit](https://pre-commit.com/) hooks for various linting tasks by installing pre-commit:

```
pre-commit install
```

3. Install `robust-llm` in developer mode, with dev dependencies:

```
pip install -e '.[dev]'
```

## Running experiments

We have various kinds of pipelines that you can run. Currently we have training pipeline (used for supervised fine-tuning of models or adversarial training), and an evaluation pipeline (used to evaluate adversarial attack on a fixed model). To choose appropriate pipeline, you need to define the `experiment.experiment_type` in a Hydra config (see below). Note that there is also a defense pipeline which is currently non-operational.

### Working with Hydra

Experiments are configured with [Hydra](https://hydra.cc/).

The config structure is defined as nested `dataclasses` in `configs.py`.
Many values are left intentionally `MISSING` so that experiments don't accidentally use defaults, and so that the final config is cleaner.

Full, runnable experiment configs are defined in `yaml` files in `robust_llm/hydra_conf`.

Experiments are run with `python robust_llm +experiment=path/to/exp.yaml` (where `path/to/exp.yaml` is relative to `robust_llm/hydra_conf/experiment`).

This setup takes inspiration from [the Hydra docs experiment example](https://hydra.cc/docs/1.3/patterns/configuring_experiments/).

The `yaml` files are set up in such a way as to be composable -- for example, we can write a `yaml` file for a model once (say `pythia-14m`) and then use that in multiple experiments and in multiple places in the config.
To do this we take advantage of [Hydra package overriding](https://hydra.cc/docs/1.0/advanced/overriding_packages/#overriding-the-package-via-the-defaults-list) (note that the linked documentation is for an older version, but is clearer than the [current version](https://hydra.cc/docs/1.3/advanced/overriding_packages/#overriding-packages-using-the-defaults-list)).

#### Package overriding
A Hydra config consists of nested _packages_: collections of values under a given path, like `training.adversarial.training_attack` or `evaluation.evaluation_attack`.
This is closely related to the concept of a _config group_, which is the path of a collection of values that Hydra is aware of.
These paths can be defined either in a directory tree of `yaml` files or in a `ConfigStore` instance in Python.

Sometimes we want to change which config group is used in a given position in the config.
As an example, let's consider setting the `training_attack` and `evaluation_attack` in an adversarial training run.
We define our attacks to be in the `attack` config group.
This is done by using `group="attack"` in `robust_llm/config/attack_configs.py` and by putting `yaml` files in the `robusts_llm/hydra_conf/attack` directory.
However, we want these configs to be placed at `training.adversarial.training_attack` and `evaluation.evaluation_attack`, _not_ at `attack`.
This requires [overriding packages](https://hydra.cc/docs/advanced/overriding_packages/) in the Defaults List.

The syntax to take an attack (say, `GCG`) from the config group `attack` and place it at `evaluation.evaluation_attack` is:
```yaml
attack@evaluation.evaluation_attack: GCG
```
However, there's an added complication: most of the time we want to write these overrides in a `yaml` script in `hydra_conf/experiment`, which is outside of the standard hierarchy.
To do this we have to use the `# @package _global_` directive, which says that the paths we define should be relative to the `hydra_conf` root, rather than relative to the current file (which would be `hydra_conf/experiment` if we didn't change anything).
With that change, we now need to put a slash in front of `attack` to indicate that Hydra should look relative to the root `hydra_conf` rather than relative to `hydra_conf/experiment`:
```yaml
/attack@evaluation.evaluation_attack: GCG
```
(TODO: work out why we don't need to write `/evaluation.evaluation_attack`)

#### Some conventions
We'll put experiments in `experiment/<ExperimentType>` directories for organization; e.g. `experiment/AdvTraining`.
(Because we're using `@package _global_`, it's fine for our experiment configs to be nested since everything is relative to the top-level config group anyway.)

In `yaml` files, we'll use `unquoted` strings to refer to other configs, and `"quoted"` strings for literal string values.
Names of other `yaml` files will be `lowercase-with-hyphens`, while defaults from python will be `UPPERCASE_WITH_UNDERSCORES` (Python defaults are explained in more detail in the worked example below).

#### Full `yaml` example
Let's look at the contents of `robust_llm/hydra_conf/experiment/Eval/pm_gcg.yaml`, which defines an evaluation experiment.
We'll go through some of the important lines.

```yaml
# @package _global_
defaults:
- /dataset: passwordmatch
- /evaluation: DEFAULT
- /attack@evaluation.evaluation_attack: GCG
- /model: EleutherAI/pythia-14m
- _self_

dataset:
    n_val: 10

experiment_name: ???
run_name: ???
experiment_type: "evaluation"
```
##### `@package _global_`
The `@package _global_` directive is important to relocate the overrides in the file.
For example, without the `@package _global_` directive, the dataset below would be placed at `args.experiment.Eval.dataset` rather than `args.dataset`, which doesn't exist (since `experiment` is not part of the config structure).

Similarly, we have to use absolute paths for the overrides below so hydra knows to look for these config files relative to the `hydra_conf` root: For example, at `dataset` rather than `experiment/Eval/dataset`.

##### `/dataset: passwordmatch`
This means to look in the `/dataset` directory (which is `hydra_conf/dataset`, since `hydra_conf` is the root) and use `passwordmatch.yaml`.

##### `/evaluation: DEFAULT`
This line is a little tricky. Strictly speaking, `/evaluation` means to look in the `/evaluation` _config group_ rather than the `/evaluation` _directory_ (and analogously for `/dataset`).
Things can end up in the `/evaluation` config group either by being in `hydra_conf/evaluation` or by being defined in Python using `hydra.ConfigStore`.

In this case, we have defined the default `evaluation` config in `configs.py`, using `cs.store(name="DEFAULT", group="evaluation", node=EvaluationConfig)`. One reason for doing this is because we rarely change most of the attributes of `EvaluationConfig` (except `evaluation_attack`, which we will handle next), and another, maybe more compelling reason is because we want to have the default value of `evaluation` in `ExperimentConfig` be `None` to avoid clogging up the config when we don't need it, but then it's tricky to specify that we want an instance of `EvaluationConfig` just from the `yaml`.

##### `/attack@evaluation.evaluation_attack: GCG`
This line says to take a config from the `/attack` config group and put it at
`evaluation.evaluation_attack`.  In particular, we want to take `/attack/GCG`
(which we know is defined in Python because it's `UPPERCASE`; in particular it's
in `config/attack_configs.py`) and set this as the `evaluation_attack`. This
uses Hydra's [package override](https://hydra.cc/docs/1.3/advanced/overriding_packages/#overriding-packages-using-the-defaults-list)
syntax.

##### `/model: EleutherAI/pythia-14m`
Analogous to the `/dataset` line: look in `/model` and use `EleutherAI/pythia-14m`.

##### `_self_`
This line is important as it tells Hydra that the stuff that comes after the [Defaults List](https://hydra.cc/docs/1.3/advanced/defaults_list/) should override the stuff _in_ the defaults list.

##### `n_val: 10`
The previous stuff was all in the [Defaults List](https://hydra.cc/docs/1.3/advanced/defaults_list/).
This line overrides whatever value `dataset.n_val` had before (which was `0`, from `configs.py`).

#### Examples of different kinds of configs
- See `robust_llm/hydra_conf/Eval/_template.yaml` for a template for `evaluation` experiments that explains some of what's going on
    - See also `_template.yaml` under `AdvTraining`, `Training`, and `DefendedEval`.
- See `random-token-n-its-1280` and `gcg-standard` in `robust_llm/hydra_conf/attack` for examples of extending/overriding the `AttackConfig` defaults.
- See `robust_llm/hydra_conf/experiment/ian/20240429_pm_random-token-fted.yaml` for an example of extending a generic experiment `yaml` for a specific experiment.
- See the scripts in `experiments/_example` for examples of how these configs could be used for real experiments.

#### Other gotchas
- On the command line or in a Python experiment script:
    - When we want to override a default value with a _config_ (like `ModelConfig`), not just a _value_ (like `model_family`), if the default value comes from the `dataclass` and was not set using the [Defaults List](https://hydra.cc/docs/advanced/defaults_list/), then we have to [prepend a `+`](https://hydra.cc/docs/1.2/advanced/override_grammar/basic/#modifying-the-defaults-list) to the override string to add it to the Defaults List.
    - An example of this is given in `experiments/_example/example_004_Eval_pm_random-token_and_gcg.py`.

### Running with `accelerate`

We use [accelerate](https://huggingface.co/docs/accelerate/en/index) for multi-GPU runs with [FSDP](https://huggingface.co/docs/accelerate/en/usage_guides/fsdp). This is because some models (e.g. 12B Pythia) are too big to do fine-tuning or run attacks against them using only 1 GPU. FSDP spreads model parameters across multiple GPUs.

To locally run any experiment with `accelerate`, instead of running

```
python robust_llm +experiment=my_exp
```

use

```
accelerate launch --config_file=accelerate_config.yaml --num_processes=<NUM_GPUS> robust_llm +experiment=my_exp
```

If you want to use FSDP with batch jobs, simply set `gpu=<NUM_GPUS>` to a number greater than 1 when calling `run_multiple()`.

*Note*: not all our code has been adapted to be used with accelerate. Things that should currently work are: fine-tuning models, adversarial evals with GCG, adversarial evals with beam search. Please use with caution.

### Running with checkpoints for preemption tolerance

The training pipeline supports checkpointing. This is controlled by `save_strategy`, `save_steps` and `save_total_limit` in the `TrainingConfig`. If more than `save_steps` are completed during a run, then a checkpoint will be saved in a directory like `trainer/run_name/checkpoint-0`. This contains the following files necessary to record the full state of the trainer:
```
config.json
model.safetensors
optimizer.pt
rng_state.pth
scheduler.pt
trainer_state.json
training_args.json
adversarial_training_state
```
All of these are handled directly by HuggingFace's Trainer class except for the last which is used to record project-specific state such as adversarial training round and attack RNG state.

If a new run is started with the same name, then we will try to find the last checkpoint in `trainer/run_name` to resume.

## Models
We currently support the following model families:
- [Gemma](https://huggingface.co/collections/google/gemma-2-release-667d6600fd5220e7b967f315)
- [Pythia](https://huggingface.co/collections/EleutherAI/pythia-scaling-suite-64fb5dfa8c21ebb3db7ad2e1)
- [Llama2](https://huggingface.co/collections/meta-llama/llama-2-family-661da1f90a9d678b6f55773b)
- [Llama3](https://huggingface.co/collections/meta-llama/meta-llama-3-66214712577ca38149ebb2b6)
- [Qwen1.5](https://huggingface.co/collections/Qwen/qwen15-65c0a2f577b1ecb76d786524)
- [Qwen2](https://huggingface.co/collections/Qwen/qwen2-6659360b33528ced941e557f).
- [Qwen2.5](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e)

You can find the relevant configs nested in robust_llm/hydra_conf/model/meta-llama/Llama-2-7b-chat-hf.yaml for example.

Note that the directory structure and names are generally chosen to mirror the names used on [HuggingFace](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).

## Datasets

Currently, we use the following datasets in our experiments:

- *IMDB*, a task to classify whether movie review is positive or negative. You can use it by setting `dataset.dataset_type="AlignmentResearch/IMDB"`
- *Spam*, a task to classify whether an email is spam or not. You can use it by setting `dataset.dataset_type="AlignmentResearch/EnronSpam"`
- *WordLength*, synthetic binary classification task to predict which of the two words is longer. You can use it by setting `dataset.dataset_type="AlignmentResearch/WordLength"`
- *PasswordMatch*, synthetic binary classification task to predict whether two passwords are identical. You can use it by setting `dataset.dataset_type="AlignmentResearch/PasswordMatch"`
- *StrongREJECT*, a jailbreak/refusal dataset from https://arxiv.org/abs/2402.10260. You can use it by setting `dataset.dataset_type="AlignmentResearch/StrongREJECT"`. This is currently our only dataset which is purely for generative models.
- *Helpful*, a human preference dataset for chatbot conversations comparing two responses of varying helpfulness. You can use by setting `dataset.dataset_type="AlignmentResearch/Helpful"`
- *Harmless*, a human preference dataset for chatbot conversations comparing two responses of varying harmlessness. You can use by setting `dataset.dataset_type="AlignmentResearch/Harmless"`.

[More info](robust_llm/rllm_datasets/README.md).

## Fine-tuned models

We store our fine-tuned models on HuggingFace. You can find up-to-date IDs of the models [here](https://docs.google.com/document/d/1fsNqlQRlv4TGJ_tK3PhFNesEmBUV2_i_EI5llsuHcFA).
