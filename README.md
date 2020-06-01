# dtt-hallucinations-disentanglement

This repo is for training a data-to-text encoder-decoder, whose decodeur is multi-branch: during decoding, it weights several RNNs to compute probability of next token.

During training, weights are conditionned based on the probability of the target token to be a divergence from the table.

During inference, weights are fixed to only use RNNs which where associated to non-diverging tokens during training.

To work work with OpenNMT-py, you will need the following libraries:

`pip install torch torchtext configargpase`

# Dataset: download & format

For all dataset related operations, please refer to `data/README.md`

You can download a file where each token has been scored, and examples are separated by empty lines

`
wget https://datacloud.di.unito.it/index.php/s/zWs2MTzB6Bxfksw/download
mv download data/
`

This file can be processed into onmt-readable file with:

`python data/format_weights.py --orig download --dest train_weights.txt --strategy binary --normalize --weight_regularization 1`


Note that you can train model on a small version of the dataset using, which can be created with the following script:

`python data/truncate_wikibio.py --folder small --max_size 1e4 --setname train test`



# Training

First things first, we compartimentalize experiments.

`python create-experiment.py --dataset wikibio --name sn3`

We use the OpenNMT-py framework for training, included in `onmt/`. Our model has been added to our version of the repo.
Training needs a preprocessing step:

`mkdir experiments/wikibio/folder_with_dataset`
`python run_onmt.py --preprocess --config preprocess.cfg`


Now that the preprocessing step is done, we can train a model using:

`python run_onmt.py --train --config train_sn3.cfg`

Please note that the model reported in our paper can be trained using the following config files: `train_sn2.cfg` and `train_sn3.cfg`


# Inference

The previous training step has saved checkpoints accross time. In order to find the best check point, you can use the following command:

`python batch_translate.py --dataset wikibio --setname valid --experiment small --bsz 64 --bms 10 --blk 0 --gpu 0 --weights 0.5 0.4 0.1`

This will generate text for every saved checkpoints with a batch size 64 and a beam size 10.


# Evaluation

### BLEU and  PARENT
We evaluate using the [BLEU](https://www.aclweb.org/anthology/P02-1040.pdf) classic metric and
[PARENT](https://www.aclweb.org/anthology/P19-1483.pdf), which his more suitable for the table-to-text generation task.
Those n-gram based metrics can be computed as follows:
```bash
python3 compute_ngram_metrics.py data/wikibio/test_tables.jl data/wikibio/test_output.txt $OUTPUT_FILE
```

### Sum of hallucination scores
We can easily generate token-level hallucination score file in the same way we did for the training data (see the
`data/README.md` file for details). Once obtained the `$OUTPUT_SCORES` file, we compute the mean of the sentence-level
sum of the hallucination scores:
```bash
python3 data/avg_hallucination_sum.py $OUTPUT_SCORES
```

### Readability tests
We report the Flesch readability test. However note, that several classic readbility tests can be performed :
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

