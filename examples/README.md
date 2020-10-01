# Examples

<<<<<<< HEAD
<<<<<<< HEAD
Version 2.9 of `transformers` introduces a new `Trainer` class for PyTorch, and its equivalent `TFTrainer` for TF 2.
=======
Version 2.9 of `transformers` introduces a new [`Trainer`](https://github.com/huggingface/transformers/blob/master/src/transformers/trainer.py) class for PyTorch, and its equivalent [`TFTrainer`](https://github.com/huggingface/transformers/blob/master/src/transformers/trainer_tf.py) for TF 2.
>>>>>>> 865d4d595eefc8cc9cee58fec9179bd182be0e2e
=======
Version 2.9 of 🤗 Transformers introduces a new [`Trainer`](https://github.com/huggingface/transformers/blob/master/src/transformers/trainer.py) class for PyTorch, and its equivalent [`TFTrainer`](https://github.com/huggingface/transformers/blob/master/src/transformers/trainer_tf.py) for TF 2.
Running the examples requires PyTorch 1.3.1+ or TensorFlow 2.2+.
>>>>>>> de4d7b004a24e4bb087eb46d742ea7939bc74644

Here is the list of all our examples:
- **grouped by task** (all official examples work for multiple models)
- with information on whether they are **built on top of `Trainer`/`TFTrainer`** (if not, they still work, they might just lack some features),
- whether they also include examples for **`pytorch-lightning`**, which is a great fully-featured, general-purpose training library for PyTorch,
- links to **Colab notebooks** to walk through the scripts and run them easily,
- links to **Cloud deployments** to be able to deploy large-scale trainings in the Cloud with little to no setup.

This is still a work-in-progress – in particular documentation is still sparse – so please **contribute improvements/pull requests.**


<<<<<<< HEAD
<<<<<<< HEAD
## Tasks built on Trainer

| Task | Example datasets | Trainer support | TFTrainer support | pytorch-lightning | Colab | One-click Deploy to Azure (wip) | 
|---|---|:---:|:---:|:---:|:---:|:---:|
| [`language-modeling`](./language-modeling) | Raw text | ✅ | - | - | - | - |
| [`text-classification`](./text-classification) | GLUE, XNLI | ✅ | ✅ | ✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/transformers/blob/master/notebooks/trainer/01_text_classification.ipynb) | [![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2FAzure%2Fazure-quickstart-templates%2Fmaster%2F101-storage-account-create%2Fazuredeploy.json) |
| [`token-classification`](./token-classification) | CoNLL NER | ✅ | ✅ | ✅ | - | - |
| [`multiple-choice`](./multiple-choice) | SWAG, RACE, ARC | ✅ | - | - | - | - |



## Other examples and how-to's

| Section | Description |
|---|---|
| [TensorFlow 2.0 models on GLUE](./text-classification) | Examples running BERT TensorFlow 2.0 model on the GLUE tasks. |
| [Running on TPUs](#running-on-tpus) | Examples on running fine-tuning tasks on Google TPUs to accelerate workloads. |
| [Language Model training](./language-modeling) | Fine-tuning (or training from scratch) the library models for language modeling on a text dataset. Causal language modeling for GPT/GPT-2, masked language modeling for BERT/RoBERTa. |
| [Language Generation](./text-generation) | Conditional text generation using the auto-regressive models of the library: GPT, GPT-2, Transformer-XL and XLNet. |
| [GLUE](./text-classification) | Examples running BERT/XLM/XLNet/RoBERTa on the 9 GLUE tasks. Examples feature distributed training as well as half-precision. |
| [SQuAD](./question-answering) | Using BERT/RoBERTa/XLNet/XLM for question answering, examples with distributed training. |
| [Multiple Choice](./multiple-choice) | Examples running BERT/XLNet/RoBERTa on the SWAG/RACE/ARC tasks. |
| [Named Entity Recognition](./token-classification) | Using BERT for Named Entity Recognition (NER) on the CoNLL 2003 dataset, examples with distributed training. |
| [XNLI](./text-classification) | Examples running BERT/XLM on the XNLI benchmark. |
| [Adversarial evaluation of model performances](./adversarial) | Testing a model with adversarial evaluation of natural language inference on the Heuristic Analysis for NLI Systems (HANS) dataset (McCoy et al., 2019.) |

## Important note

**Important**
To make sure you can successfully run the latest versions of the example scripts, you have to install the library from source and install some example-specific requirements.
Execute the following steps in a new virtual environment:

```bash
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
pip install -r ./examples/requirements.txt
```

## Running on TPUs

Documentation to come.
=======
# The Big Table of Tasks
=======
## The Big Table of Tasks
>>>>>>> de4d7b004a24e4bb087eb46d742ea7939bc74644

| Task | Example datasets | Trainer support | TFTrainer support | pytorch-lightning | Colab
|---|---|:---:|:---:|:---:|:---:|
| [**`language-modeling`**](https://github.com/huggingface/transformers/tree/master/examples/language-modeling)       | Raw text        | ✅ | -  | -  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb)
| [**`text-classification`**](https://github.com/huggingface/transformers/tree/master/examples/text-classification)   | GLUE, XNLI      | ✅ | ✅ | ✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/trainer/01_text_classification.ipynb)
| [**`token-classification`**](https://github.com/huggingface/transformers/tree/master/examples/token-classification) | CoNLL NER       | ✅ | ✅ | ✅ | -
| [**`multiple-choice`**](https://github.com/huggingface/transformers/tree/master/examples/multiple-choice)           | SWAG, RACE, ARC | ✅ | ✅ | -  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ViktorAlm/notebooks/blob/master/MPC_GPU_Demo_for_TF_and_PT.ipynb)
| [**`question-answering`**](https://github.com/huggingface/transformers/tree/master/examples/question-answering)     | SQuAD           | ✅ | ✅ | -  | -
| [**`text-generation`**](https://github.com/huggingface/transformers/tree/master/examples/text-generation)           | -               | n/a | n/a | n/a | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/02_how_to_generate.ipynb)
| [**`distillation`**](https://github.com/huggingface/transformers/tree/master/examples/distillation)                 | All             | - | -  | -  | -
| [**`summarization`**](https://github.com/huggingface/transformers/tree/master/examples/seq2seq)                     | CNN/Daily Mail  | ✅  | -  | ✅  | -
| [**`translation`**](https://github.com/huggingface/transformers/tree/master/examples/seq2seq)                       | WMT             | ✅  | -  | ✅  | -
| [**`bertology`**](https://github.com/huggingface/transformers/tree/master/examples/bertology)                       | -               | - | -  | -  | -
| [**`adversarial`**](https://github.com/huggingface/transformers/tree/master/examples/adversarial)                   | HANS            | ✅ | -  | -  | -


<br>

## Important note

**Important**
To make sure you can successfully run the latest versions of the example scripts, you have to install the library from source and install some example-specific requirements.
Execute the following steps in a new virtual environment:

```bash
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
pip install -r ./examples/requirements.txt
```

## One-click Deploy to Cloud (wip)

**Coming soon!**

## Running on TPUs

When using Tensorflow, TPUs are supported out of the box as a `tf.distribute.Strategy`.

When using PyTorch, we support TPUs thanks to `pytorch/xla`. For more context and information on how to setup your TPU environment refer to Google's documentation and to the
very detailed [pytorch/xla README](https://github.com/pytorch/xla/blob/master/README.md).

In this repo, we provide a very simple launcher script named [xla_spawn.py](https://github.com/huggingface/transformers/tree/master/examples/xla_spawn.py) that lets you run our example scripts on multiple TPU cores without any boilerplate.
Just pass a `--num_cores` flag to this script, then your regular training script with its arguments (this is similar to the `torch.distributed.launch` helper for torch.distributed).

For example for `run_glue`:

```bash
python examples/xla_spawn.py --num_cores 8 \
	examples/text-classification/run_glue.py
	--model_name_or_path bert-base-cased \
	--task_name mnli \
	--data_dir ./data/glue_data/MNLI \
	--output_dir ./models/tpu \
	--overwrite_output_dir \
	--do_train \
	--do_eval \
	--num_train_epochs 1 \
	--save_steps 20000
```

Feedback and more use cases and benchmarks involving TPUs are welcome, please share with the community.
<<<<<<< HEAD
>>>>>>> 865d4d595eefc8cc9cee58fec9179bd182be0e2e
=======

## Logging & Experiment tracking

You can easily log and monitor your runs code. The following are currently supported:

* [TensorBoard](https://www.tensorflow.org/tensorboard)
* [Weights & Biases](https://docs.wandb.com/library/integrations/huggingface)
* [Comet ML](https://www.comet.ml/docs/python-sdk/huggingface/)

### Weights & Biases

To use Weights & Biases, install the wandb package with:

```bash
pip install wandb
```

Then log in the command line:

```bash
wandb login
```

If you are in Jupyter or Colab, you should login with:

```python
import wandb
wandb.login()
```

Whenever you use `Trainer` or `TFTrainer` classes, your losses, evaluation metrics, model topology and gradients (for `Trainer` only) will automatically be logged.

When using 🤗 Transformers with PyTorch Lightning, runs can be tracked through `WandbLogger`. Refer to related [documentation & examples](https://docs.wandb.com/library/integrations/lightning).

### Comet.ml

To use `comet_ml`, install the Python package with:

```bash
pip install comet_ml
```

or if in a Conda environment:

```bash
conda install -c comet_ml -c anaconda -c conda-forge comet_ml
```
>>>>>>> de4d7b004a24e4bb087eb46d742ea7939bc74644
