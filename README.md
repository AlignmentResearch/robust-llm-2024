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

### Running ad-hoc experiments locally / on a devbox

Experiments are configured with [Hydra](https://hydra.cc/). You can run the default configuration with:

```
python robust_llm
```

All config options are defined in `robust_llm/configs.py`. Defaults can be overridden via the command line, e.g.:

```
python robust_llm experiment.experiment_type=training experiment.environment.seed=42
```

or

```
python robust_llm experiment.experiment_type=evaluation experiment.environment.dataset_type=hf/imdb experiment.evaluation.evaluation_attack.attack_type=search_based
```

Alternatively, you can also define new config files, which is the recommended strategy for saving experiment configurations long term. You can see example config files in /robust_llm/hydra_conf/experiment. If you add a new file to /robust_llm/hydra_conf/experiment called `my_exp.yaml` then you can use it with:

```
python robust_llm +experiment=my_exp
```

Note that you can use a new config file as a starting point (for example, for a collection of experiments) and then adjust individual options in the same way as above, for example:

`python robust_llm +experiment=my_exp experiment.environment.seed=43`

The complete configuration used will be printed as `Configuration arguments:`. This description can be copied into a new file in the /robust_llm/hydra_conf/experiment directory if you want to repeat it later.

### Running a single batch job

If you have your hydra config prepared and want to run an experiment on the cluster in batch mode, you can use the `run_batch_job.py` script. To use it, make sure you check out the correct git commit that you want in your experiment, and that all the relevant changes are committed. The container will first set up a code directory with the commit matching your current repo, and then run the experiment. Usage:

```
python run_batch_job.py --hydra_config=<HYDRA_CONFIG_NAME> [--experiment_name=<EXP_NAME> --job_type=<JOB_TYPE> --container_tag=<TAG>]
```

As a requirement, you have to set up `docker` ([instructions](https://github.com/AlignmentResearch/flamingo/wiki/Docker-tutorial:-secure-credentials-and-basic-use#read-only-credentials-for-your-cluster-account)), `github-credentials` ([instructions](https://github.com/AlignmentResearch/flamingo/wiki/Build-Docker-images-on-the-cluster:-Kaniko#authentication-1-pulling-from-your-private-github-repo)), `wandb`, and `huggingface` kubernetes secrets.

For `wandb`, use the following command:

```
kubectl create secret generic wandb --from-literal=api-key=<YOUR_WANDB_API_KEY>
```

For `huggingface`, use the following command:

```
kubectl create secret generic huggingface --from-literal=token=<YOUR_HF_TOKEN>
```

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

If you want to use FSDP with batch jobs, simply specify `use_accelerate=True` and `gpu=<NUM_GPUS>` as arguments of `run_multiple()`.

*Note*: not all our code has been adapted to be used with accelerate. Things that should currently work are: fine-tuning models, adversarial evals with GCG, adversarial evals with beam search. Please use with caution.

## Datasets

Currently, we use the following datasets in our experiments:

- *tensor_trust*, synthetic binary classification task, where the label is True iff two passwords given in the input match. You can use it by setting `environment.dataset_type="tensor_trust"`
- *IMDB*, a task to classify whether movie review is positive or negative. You can use it by setting `environment.dataset_type="hf/imdb"`
- *spam*, a task to classify whether an email is spam or not. You can use it by setting `environment.dataset_type="hf/SetFit/enron_spam"`
- *word_length*, synthetic binary classification task to predict which of the two words is longer.

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
