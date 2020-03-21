# dtt-hallucinations-disentanglement

This repo is for training a data-to-text encoder-decoder, whose decodeur is multi-branch: during decoding, it weights several RNNs to compute probability of next token.

During training, weights are conditionned based on the probability of the target token to be a divergence from the table.

During inference, weights are fixed to only use RNNs which where associated to non-diverging tokens during training.

# Dataset: download & format

For all dataset related operations, please refer to `data/README.md`

# Training

First things first, we compartimentalize experiments.

`python create_experiment.py --dataset wikibio --name small`

We use the OpenNMT-py framework for training, included in `onmt/`. Our model has been added to our version of the repo.
Training needs a preprocessing step:

`python run_onmt.py --preprocess --config preprocess.cfg`

Note that for now, the `preprocess.cfg` is setup to find a small version of the dataset, which can be created with the following script:

`python data/truncate_wikibio.py --folder small --max_size 1e4 --setname train test`

Now that the preprocessing step is done, we can train a model using:

`python run_onmt.py --train --config train.cfg`