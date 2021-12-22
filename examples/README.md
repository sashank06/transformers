<<<<<<< HEAD
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
=======
<!---
Copyright 2020 The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
>>>>>>> 8d43c71a1ca3ad322cc45008eb66a5611f1e017e

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Examples

<<<<<<< HEAD
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
=======
This folder contains actively maintained examples of use of 🤗 Transformers organized along NLP tasks. If you are looking for an example that used to be in this folder, it may have moved to the corresponding framework subfolder (pytorch, tensorflow or flax), our [research projects](https://github.com/huggingface/transformers/tree/master/examples/research_projects) subfolder (which contains frozen snapshots of research projects) or to the [legacy](https://github.com/huggingface/transformers/tree/master/examples/legacy) subfolder.
>>>>>>> 8d43c71a1ca3ad322cc45008eb66a5611f1e017e

While we strive to present as many use cases as possible, the scripts in this folder are just examples. It is expected that they won't work out-of-the box on your specific problem and that you will be required to change a few lines of code to adapt them to your needs. To help you with that, most of the examples fully expose the preprocessing of the data. This way, you can easily tweak them.

This is similar if you want the scripts to report another metric than the one they currently use: look at the `compute_metrics` function inside the script. It takes the full arrays of predictions and labels and has to return a dictionary of string keys and float values. Just change it to add (or replace) your own metric to the ones already reported.

Please discuss on the [forum](https://discuss.huggingface.co/) or in an [issue](https://github.com/huggingface/transformers/issues) a feature you would like to implement in an example before submitting a PR: we welcome bug fixes but since we want to keep the examples as simple as possible, it's unlikely we will merge a pull request adding more functionality at the cost of readability.

## Important note

**Important**

To make sure you can successfully run the latest versions of the example scripts, you have to **install the library from source** and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:
```bash
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
```
Then cd in the example folder of your choice and run
```bash
pip install -r requirements.txt
```

<<<<<<< HEAD
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

=======
To browse the examples corresponding to released versions of 🤗 Transformers, click on the line below and then on your desired version of the library:

<details>
  <summary>Examples for older versions of 🤗 Transformers</summary>
	<ul>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.5.1/examples">v4.5.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.4.2/examples">v4.4.2</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.3.3/examples">v4.3.3</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.2.2/examples">v4.2.2</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.1.1/examples">v4.1.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.0.1/examples">v4.0.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.5.1/examples">v3.5.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.4.0/examples">v3.4.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.3.1/examples">v3.3.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.2.0/examples">v3.2.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.1.0/examples">v3.1.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.0.2/examples">v3.0.2</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.11.0/examples">v2.11.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.10.0/examples">v2.10.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.9.1/examples">v2.9.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.8.0/examples">v2.8.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.7.0/examples">v2.7.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.6.0/examples">v2.6.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.5.1/examples">v2.5.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.4.0/examples">v2.4.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.3.0/examples">v2.3.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.2.0/examples">v2.2.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.1.0/examples">v2.1.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.0.0/examples">v2.0.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v1.2.0/examples">v1.2.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v1.1.0/examples">v1.1.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v1.0.0/examples">v1.0.0</a></li>
	</ul>
</details>

<<<<<<< HEAD
Alternatively, you can find switch your cloned 🤗 Transformers to a specific version (for instance with v3.5.1) with
>>>>>>> 8d43c71a1ca3ad322cc45008eb66a5611f1e017e
=======
Alternatively, you can switch your cloned 🤗 Transformers to a specific version (for instance with v3.5.1) with
>>>>>>> d0422de5634d3b14ac5159d7f8a2ab9336821d22
```bash
git checkout tags/v3.5.1
```
<<<<<<< HEAD
>>>>>>> de4d7b004a24e4bb087eb46d742ea7939bc74644
=======
and run the example command as usual afterward.
>>>>>>> 8d43c71a1ca3ad322cc45008eb66a5611f1e017e
