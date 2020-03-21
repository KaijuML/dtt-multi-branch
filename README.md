# dtt-hallucinations-disentanglement

This repo is for training a data-to-text encoder-decoder, whose decodeur is multi-branch: during decoding, it weights several RNNs to compute probability of next token.
During training, weights are conditionned based on the probability of the target token to be a divergence from the table.
During inference, weights are fixed to only use RNNs which where associated to non-diverging tokens during training.

# Dataset: download & format

For all dataset related operations, please refer to `data/README.md`

# Training

First things first, we create compartimentalize experiments.

`python create_experiment.py --dataset wikibio --name small`

We use the OpenNMT-py framework for training, included in `onmt/`. Our model has been added to our version of the repo.
Training needs a preprocessing step:

```python
python run_onmt.py --preprocess --config preprocess.cfg
```

For now, the `preprocess.cfg` is setup to find a small version of the dataset, which can be created with the following script:

```python
import os

def truncate_wikibio_files(setname, length=10):
    assert setname in ['train', 'valid', 'test']
    assert int(length) == length and length > 0  # check positiv int
    
    for suffix in ['input', 'output', 'weights']:
        orig = os.path.join('data', 'wikibio', f'{setname}_{suffix}.txt')
        dest = os.path.join('data', 'wikibio', 'small', f'{setname}_{suffix}.txt')
        with open(orig, mode="r", encoding='utf8') as f, \
                open(dest, mode="w", encoding='utf8') as g:
            for idx, line in enumerate(f):
                if idx > length:
                    break
                g.write(line)

folder = os.path.join('wikibio', 'data', 'small')
if not os.path.exists(folder): os.mkdir(folder)
    
# We take the first ten thousands
truncate_wikibio_files('train', int(1e4))
truncate_wikibio_files('test', int(1e4))
```

Now that the preprocessing step is down, we can train a model using:

`python run_onmt.py --train --config train.cfg`