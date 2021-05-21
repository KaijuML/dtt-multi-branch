# Controlling Hallucinations at Word Level in Data-to-Text Generation

Code for our paper [Controlling Hallucinations at Word Level in Data-to-Text Generation](https://arxiv.org/abs/2102.02810) (Clément Rebuffel, Marco Roberti, Laure Soulier, Geoffrey Scoutheeten, Rossella Cancelliere, Patrick Gallinari).

With this code, you train a data-to-text encoder-decoder, whose decoder is multi-branch: during decoding, it weights several RNNs to compute probability of next token.

  1) During training, weights are conditioned based on the probability of the target token to be a divergence from the table.  
  2) During inference, weights are fixed to only use RNNs which where associated to non-diverging tokens during training.
  
We also provide preprocessing scripts that work on WikiBio and ToTTo; as well as evaluation scripts for all metrics reported in the paper.

Note that most command are given for WikiBio, but will also work for ToTTo. If something is not straight forward, please open an issue or create a discussion.

### Requirements

To work work with OpenNMT-py, you will need the following libraries:

```bash
pip3 install torch torchtext configargpase
```


## Dataset: download & format

For all dataset related operations, please refer to `data/README.md`

We provide already an processed file for the WikiBio dataset: You can download a file where each line is a scored token, and examples are separated by empty lines

```bash
wget https://datacloud.di.unito.it/index.php/s/KPr9HnbMyNWqRdj/download
mv download data/
```

This file can be processed into OpenNMT-readable file with:

```bash
python3 data/format_weights.py --orig data/download --dest train_weights.txt --strategy thresholds --thresholds 0.4 --normalize --weight_regularization 1 --eos_weights 1 0
```

This will use a fixed weight for the fluency factor (called in the script `weight_regularization`) and will give token to the content branch if they are scored below 0.4, else to the hallucination branch. `--normalize` means weights are normalized by their sum (so that they sum to 1) and `--eos_weights 1 0` means that the End-of-Sequence token will be the responsability of the content branch.


Note that if you want to train a model on a small version of the dataset for practical reasons, you can create it with the following script:

```bash
python3 data/truncate_wikibio.py --folder small --max_size 1e4 --setname train test
```



## Training

First things first, we compartmentalize experiments.

```bash
python3 create-experiment.py --dataset wikibio --name mbd
```

An experiment folder is now available at `experiments/wikibio/mbd`. You can move the previously created weights there:

```bash
mv train_weights.txt experiments/wikibio/mbd/train_weights.txt
```

We use the OpenNMT-py framework for training, included in `onmt/`. Our model has been added to our version of the repo.
Training needs a preprocessing step:

```bash
mkdir experiments/wikibio/folder_with_dataset
python3 run_onmt.py --preprocess --config preprocess.cfg
```


Now that the preprocessing step is done, we can train a model using:

```bash
python3 run_onmt.py --train --config train_mbd.cfg
```

Please note that the model reported in our paper can be trained using the `train_mbd.cfg` config file.


## Inference

The previous training step has saved checkpoints across time. In order to find the best check point, you can use the following command:

```bash
python3 batch_translate.py --dataset wikibio --setname valid --experiment small --bsz 64 --bms 10 --blk 0 --gpu 0 --weights 0.5 0.4 0.1
```

This will generate text for every saved checkpoints with a batch size 64 and a beam size 10.


## Evaluation

### BLEU and  PARENT
We evaluate using the [BLEU](https://www.aclweb.org/anthology/P02-1040.pdf) classic metric and
[PARENT](https://www.aclweb.org/anthology/P19-1483.pdf), which is more suitable for the table-to-text generation task.
Those n-gram based metrics can be computed as follows:
```bash
python3 compute_ngram_metrics.py data/wikibio/test_tables.jl data/wikibio/test_output.txt $OUTPUT_FILE
```

### Hallucination rate
We can easily generate token-level hallucination score file in the same way we did for the training data (see the
`data/README.md` file for details). Once obtained the `$OUTPUT_SCORES` file, we compute the mean of the sentence-level
hallucination rates, considering as hallucinated all tokens that are above a given threshold:
```bash
python3 data/avg_hallucination_rate.py $OUTPUT_SCORES
```

### Readability tests
We report the Flesch readability test. However note, that several classic readability tests can be performed :
 * [Kincaid](https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests)
 * [ARI](https://en.wikipedia.org/wiki/Automated_readability_index)
 * [Coleman-Liau](https://en.wikipedia.org/wiki/Coleman%E2%80%93Liau_index)
 * [Flesch Index](https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests)
 * [Fog Index](https://en.wikipedia.org/wiki/Gunning_fog_index)
 * [Lix](https://en.wikipedia.org/wiki/Lix_(readability_test))
 * [SMOG-Grading](https://en.wikipedia.org/wiki/SMOG)

Such values are computed using the [GNU diction/style tool](https://www.gnu.org/software/diction/) as follows:
```bash
cat $OUTPUT_FILE | sed -e 's/-lrb-/(/' -e 's/-rrb-/)/' -e 's/--/-/' -e "s/''/\"/" -e 's/``/"/' -e 's/./\u&/' | style
```

