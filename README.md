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

If you would like to run TextAttack attacks that depend on TensorFlow embeddings [such as TextFooler](https://textattack.readthedocs.io/en/latest/0_get_started/installation.html#optional-dependencies), then run:

```
pip install -e '.[dev,tensorflow]'
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
- See `random-token-n-its-1280` and `trl-stronger-adversary` in `robust_llm/hydra_conf/attack` for examples of extending/overriding the `AttackConfig` defaults.
- See `robust_llm/hydra_conf/experiment/ian/20240429_pm_random-token-fted.yaml` for an example of extending a generic experiment `yaml` for a specific experiment.
- See the scripts in `experiments/_example` for examples of how these configs could be used for real experiments.

#### Other gotchas
- On the command line or in a Python experiment script:
    - When we want to override a default value with a _config_ (like `ModelConfig`), not just a _value_ (like `model_family`), if the default value comes from the `dataclass` and was not set using the [Defaults List](https://hydra.cc/docs/advanced/defaults_list/), then we have to [prepend a `+`](https://hydra.cc/docs/1.2/advanced/override_grammar/basic/#modifying-the-defaults-list) to the override string to add it to the Defaults List.
    - An example of this is given in `experiments/_example/example_003_Eval_imdb_trl-different-adversaries.py`.

### Running a single batch job

If you have your hydra config prepared and want to run an experiment on the cluster in batch mode, you can use the `run_batch_job.py` script. To use it, make sure you check out the correct git commit that you want in your experiment, and that all the relevant changes are committed. The container will first set up a code directory with the commit matching your current repo, and then run the experiment. Usage:

```
python run_batch_job.py --hydra_config=<HYDRA_CONFIG_NAME> [--experiment_name=<EXP_NAME> --job_type=<JOB_TYPE> --container_tag=<TAG>]
```

As a requirement, you have to set up `docker` ([instructions](https://github.com/AlignmentResearch/flamingo/wiki/Docker-tutorial:-secure-credentials-and-basic-use#read-only-credentials-for-your-cluster-account)), `github-credentials` ([instructions](https://github.com/AlignmentResearch/flamingo/wiki/Build-Docker-images-on-the-cluster:-Kaniko#authentication-1-pulling-from-your-private-github-repo)), `wandb`, and `huggingface` kubernetes secrets.

For `wandb`, [get your API key](https://wandb.ai/authorize) and run the following command:

```
kubectl create secret generic wandb --from-literal=api-key=<YOUR_WANDB_API_KEY>
```

For `huggingface`, [create a read-only or fine-grained access token](https://huggingface.co/settings/tokens). For a fine-grained access token, enable "Read access to contents of all public gated repos you can access". Then run the following command:
```
kubectl create secret generic huggingface --from-literal=token=<YOUR_HF_TOKEN>
```

If you want to run `ScoringCallback`s based on `StrongREJECT` or other parts of the codebase that use the OpenAI API, you'll need to create an OpenAI API key. Ask an admin for access to the OpenAI organization and make a project API key with `Write` access to `Model capabilities`. Then run the following command:
```
kubectl create secret generic openai-api-key --from-literal=key=<YOUR_OPENAI_API_KEY>
```

You also need to ask a Flamingo admin to give you access to the `robust-llm` drive on the cluster.

### Running multiple batch jobs at once

For defining "serious" experiments that contain multiple k8s jobs inside (e.g. grid search), we use Python
files that are stored under `experiments/` directory.

For example, take a look at `experiments/mz/experiments/mz/mz_007_pythia_adv_eval_ta_tensor_trust.py`. We use the following naming convention:

- experiments by Jay Doe go into an `experiments/jd` or `experiments/jay` directory (choose one and stick with it),
- the experiment name is `jd_<NUMBER>_<SHORT_DESCRIPTION>`, where `NUMBER` is manually
increased; prepend with zeros until the number has three digits, so that the experiments are shown in order when sorted lexicographically (for example, 002, 031, or 155),
- we commit experiments to the repository. Once committed, the experiment should not be
modified under normal circumstances. This way we maintain a history of experiments.

To run an experiment (i.e. schedule multiple jobs on a k8s cluster), just run the Python file. **Please do not just randomly run other people's configs; it can mess up their wandb structure.** If you want to run a similar experiment, just copy the config into your directory, rename, modify, and run.

The experiments are defined by starting with some Hydra config and then modifying it with dictionaries with overrides.

If you want to just see the generated configs and not actually run the experiments on cluster, add `dry_run=True` to the `run_multiple()` call inside the Python file that defines the experiment.

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

## Datasets

Currently, we use the following datasets in our experiments:

- *tensor_trust*, synthetic binary classification task, where the label is True iff two passwords given in the input match. You can use it by setting `environment.dataset_type="tensor_trust"`
- *IMDB*, a task to classify whether movie review is positive or negative. You can use it by setting `environment.dataset_type="AlignmentResearch/IMDB"`
- *spam*, a task to classify whether an email is spam or not. You can use it by setting `environment.dataset_type="AlignmentResearch/EnronSpam"`
- *word_length*, synthetic binary classification task to predict which of the two words is longer. You can use it by setting `environment.dataset_type="AlignmentResearch/WordLength"`
- *Helpful*, a human preference dataset for chatbot conversations comparing two responses of varying helpfulness. You can use by setting `environment.dataset_type="AlignmentResearch/Helpful"`
- *Harmless*, a human preference dataset for chatbot conversations comparing two responses of varying harmlessness. You can use by setting `environment.dataset_type="AlignmentResearch/Harmless"`.

### Dataset-attack compatibility

"Standard natural language tasks" (IMDB, spam) are compatible with GCG, beam search, random token, TRL*. Make sure to set `experiment.evaluation.evaluation_attack.append_to_modifiable_chunk=True` when running those combinations (with this setting, the attack will just append some tokens at the end instead of removing the whole text and then adding tokens).
On IMDB, also TextFooler can be run. (We could try on spam as well but it is not clear how much the semantics of the text would change -- and hence, if the attacks will be correct most of the time.)

"Algorithmic tasks" (tensor_trust, word_length) are compatible with GCG, beam search, random token, TRL*. Make sure to set `experiment.evaluation.evaluation_attack.append_to_modifiable_chunk=False` when running those combinations. (For those datasets, there are defined small parts of text which will be replaced with attack tokens).

*TRL: Note that this algorithm's performance is still unstable in many cases.

### Tomita [DEPRECATED]

We currently do not use Tomita.

Tomita datasets must be pregenerated as files in order to be used in training.

They can be generated by running `robust_llm/dataset_management/tomita/tomita_dataset_generator.py`. You can edit the file's `__main__` function call to extend the range of examples to generate for training.

## Fine-tuned models

We store our fine-tuned models on HuggingFace. You can find up-to-date IDs of the models [here](https://docs.google.com/document/d/1fsNqlQRlv4TGJ_tK3PhFNesEmBUV2_i_EI5llsuHcFA).

## Logs

We store experiments logs in wandb under the [farai/robust-llm](https://wandb.ai/farai/robust-llm) project. We have been using some ad-hoc Colab code to process results beyond just standard wandb plots if needed; see e.g. [here](https://colab.research.google.com/drive/1tZdK1k4hZMZHZxY07ahY4vFt1X8GoLtK).

## Building Docker images

Docker images of the repo can be built using Kaniko. The first time you do so, you must follow the setup described [here](https://github.com/AlignmentResearch/flamingo/wiki/Build-Docker-images-on-the-cluster:-Kaniko), particularly setting up the Kubernetes secrets named `docker` and `github-credentials`.

To build a Docker image from the `main` branch at head, you can run:

```
kubectl create -f k8s/kaniko-build.yaml
```

If you wish to build a Docker image from a different branch, you should edit the `BRANCH_NAME` value in `k8s/kaniko-build.yaml` and then run the command above. More details can be found in the Flamingo wiki article on Kaniko, and the Kaniko docs themselves.

## Working in a devbox
The most convenient way to create a devbox is to run `make devbox` which will use `k8s/auto-devbox.yaml` and you can pass various arguments to this, e.g. CPU, GPU and MEMORY (see the Makefile).

You can also run `kubectl create -f k8s/devbox.yaml`.

Using the VSCode Kubernetes extension, you can then right click the pod and select "Attach VS Code".

It is recommended to add extensions in your user settings JSON so that they are loaded automatically, e.g. in "~/.config/Code/User/settings.json" in Ubuntu, add
```
"dev.containers.defaultExtensionsIfInstalledLocally": [
    "ms-python.debugpy",
    "ms-python.python"
]
```
