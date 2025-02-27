
## Language model training

<<<<<<< HEAD
Based on the script [`run_language_modeling.py`](https://github.com/huggingface/transformers/blob/master/examples/run_language_modeling.py).
=======
Based on the script [`run_language_modeling.py`](https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_language_modeling.py).
>>>>>>> 865d4d595eefc8cc9cee58fec9179bd182be0e2e

Fine-tuning (or training from scratch) the library models for language modeling on a text dataset for GPT, GPT-2, BERT, DistilBERT and RoBERTa. GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT, DistilBERT and RoBERTa
are fine-tuned using a masked language modeling (MLM) loss.

Before running the following example, you should get a file that contains text on which the language model will be
trained or fine-tuned. A good example of such text is the [WikiText-2 dataset](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/).

We will refer to two different files: `$TRAIN_FILE`, which contains text for training, and `$TEST_FILE`, which contains
text that will be used for evaluation.

### GPT-2/GPT and causal language modeling

The following example fine-tunes GPT-2 on WikiText-2. We're using the raw WikiText-2 (no tokens were replaced before
the tokenization). The loss here is that of causal language modeling.

```bash
export TRAIN_FILE=/path/to/dataset/wiki.train.raw
export TEST_FILE=/path/to/dataset/wiki.test.raw

python run_language_modeling.py \
    --output_dir=output \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE
```

This takes about half an hour to train on a single K80 GPU and about one minute for the evaluation to run. It reaches
a score of ~20 perplexity once fine-tuned on the dataset.

### RoBERTa/BERT/DistilBERT and masked language modeling

The following example fine-tunes RoBERTa on WikiText-2. Here too, we're using the raw WikiText-2. The loss is different
as BERT/RoBERTa have a bidirectional mechanism; we're therefore using the same loss that was used during their
pre-training: masked language modeling.

In accordance to the RoBERTa paper, we use dynamic masking rather than static masking. The model may, therefore, converge
slightly slower (over-fitting takes more epochs).

We use the `--mlm` flag so that the script may change its loss function.

If using whole-word masking, use both the`--mlm` and `--wwm` flags.

```bash
export TRAIN_FILE=/path/to/dataset/wiki.train.raw
export TEST_FILE=/path/to/dataset/wiki.test.raw

python run_language_modeling.py \
    --output_dir=output \
    --model_type=roberta \
    --model_name_or_path=roberta-base \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm \
    --wwm
```

For Chinese models, it's same with English model with only --mlm`. If using whole-word masking, we need to generate a reference files, case it's char level.

**Q :** Why ref file ?

**A :** Suppose we have a Chinese sentence like : `我喜欢你` The original Chinese-BERT will tokenize it as `['我','喜','欢','你']` in char level.
Actually, `喜欢` is a whole word. For whole word mask proxy, We need res like `['我','喜','##欢','你']`.
So we need a ref file to tell model which pos of BERT original token should be added `##`.

**Q :** Why LTP ?

**A :** Cause the best known Chinese WWM BERT is [Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm) by HIT. It works well on so many Chines Task like CLUE (Chinese GLUE).
They use LTP, so if we want to fine-tune their model, we need LTP.

```bash
export TRAIN_FILE=/path/to/dataset/wiki.train.raw
export LTP_RESOURCE=/path/to/ltp/tokenizer
export BERT_RESOURCE=/path/to/bert/tokenizer
export SAVE_PATH=/path/to/data/ref.txt

python chinese_ref.py \
    --file_name=$TRAIN_FILE \
    --ltp=$LTP_RESOURCE
    --bert=$BERT_RESOURCE \
    --save_path=$SAVE_PATH 
```
Now Chinese Ref is only supported by `LineByLineWithRefDataset` Class, so we need add `line_by_line` flag: 


```bash
export TRAIN_FILE=/path/to/dataset/wiki.train.raw
export TEST_FILE=/path/to/dataset/wiki.test.raw
export REF_FILE=/path/to/ref.txt

python run_language_modeling.py \
    --output_dir=output \
    --model_type=roberta \
    --model_name_or_path=roberta-base \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --chinese_ref_file=$REF_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm \
    --line_by_line \
    --wwm
```

### XLNet and permutation language modeling

XLNet uses a different training objective, which is permutation language modeling. It is an autoregressive method 
to learn bidirectional contexts by maximizing the expected likelihood over all permutations of the input 
sequence factorization order.

We use the `--plm_probability` flag to define the ratio of length of a span of masked tokens to surrounding 
context length for permutation language modeling.

The `--max_span_length` flag may also be used to limit the length of a span of masked tokens used 
for permutation language modeling.

```bash
export TRAIN_FILE=/path/to/dataset/wiki.train.raw
export TEST_FILE=/path/to/dataset/wiki.test.raw

python run_language_modeling.py \
    --output_dir=output \
    --model_name_or_path=xlnet-base-cased \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
```
