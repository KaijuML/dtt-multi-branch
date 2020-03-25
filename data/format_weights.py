from overrides import overrides
from tqdm import tqdm

import pkg_resources
import argparse
import os


def _count_examples(path):
    """
    In PoS/Score tagged files, it's one token per line,
    examples separated by empty line.
    """
    nlines = 0
    with open(path, mode='r', encoding='utf8') as f:
        for line in f: 
            if not line.strip(): nlines += 1
    return nlines


def read_tagged_file(path):
    """
    Accumulate lines until empty line, then yield and do it again
    """
    sentence = list()
    with open(path, mode='r', encoding='utf8') as f:
        for line in f:
            if line.strip():
                sentence.append(line.strip().split())
            else:
                yield sentence
                sentence = list()
    if sentence: 
        yield sentence


class Strategy:
    """
    Base object to convert hallucination weights to branch scores for the network.
    ARGS:
        eos_weights (float OR int): the weights used to predict </s>
        normalize (bool): whether to devide all weights by the total sum
        weight_regularization (float OR int): if >0 we add an other branch
                which will always be active, to regularize training.
                This branch is always considered the first branch (index 0)
                
    In order to implement a new strategy, only the method `_score_weight` should be
    implemented. It should be a function (float --> List[float OR int])    
    """
    def __init__(self, eos_weights, normalize=False, weight_regularization=0):
        self.normalize = normalize
        self.weight_regularization = weight_regularization

        assert all(isinstance(w, (float, int)) for w in eos_weights)
        if weight_regularization > 0:
            eos_weights.append(weight_regularization)
        if normalize:
            eos_weights = [w / sum(eos_weights) for w in eos_weights]
        self.eos_weights = eos_weights

    @staticmethod
    def chain_scores(scores):
        return ' '.join(':'.join(map(str, s)) for s in scores)

    def _score_weight(self, w):
        raise NotImplementedError

    def score_weight(self, w):
        s = [self.weight_regularization] if self.weight_regularization > 0 else list()
        s.extend(self._score_weight(w))
        if self.normalize:
            tot_s = sum(s)
            s = [_w/tot_s for _w in s]
        return s

    def format_instance(self, instance):
        """instanceinstance a List[tuple(tok=str, w=str)] of weighted tokens"""
        scores = [self.score_weight(w) for _, w in instance]
        scores.append(self.eos_weights)
        return self.chain_scores(scores)


class BinaryStrategy(Strategy):
    @overrides
    def _score_weight(self, w):
        """Tokens are either hallucinated or they're not."""
        return [1, 0] if w=='0' else [0, 1]


class OneBranchStrategy(Strategy):
    @overrides
    def __init__(self, eos_weights, normalize, weight_regularization):
        super().__init__(eos_weights, normalize, weight_regularization)
        if not self.eos_weights == [1]:
            raise ValueError('OneBranch strategy should have only one branch, '
                             'and predict </s> with weight 1. '
                             f'Instead, got {self.eos_weights} for </s> prediction.')
        if not self.weight_regularization == 0:
            raise ValueError('OneBranch strategy do not have a regularization branch!')

    @overrides
    def _score_weight(self, w):
        return [1]


strategies = {
    'binary': BinaryStrategy,
    'one_branch': OneBranchStrategy,
}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--orig', '-o', dest='orig',
                        help='Name of the weights-tagged examples')
    parser.add_argument('--dest', '-d', dest='dest',
                        help='Name of the resulting file')
    
    parser.add_argument('--strategy', '-s', dest='strategy',
                        choices=list(strategies), default="binary",
                        help='Strategy to format weights.')
    
    parser.add_argument('--normalize', dest='normalize', action='store_true',
                        help='Divide individual weights by the total sum.')
    
    parser.add_argument('--weight_regularization', dest='weight_regularization',
                        type=float, default=0,
                        help='Weights for regularization branch. '
                             'Zero means no branch.')
    
    parser.add_argument('--eos_weights', dest='eos_weights', nargs='+',
                        type=float, help='Weights to predict </s> token.')

    args = parser.parse_args()

    folder = pkg_resources.resource_filename(__name__, 'wikibio')
    orig = os.path.join(folder, args.orig)
    dest = os.path.join(folder, args.dest)

    assert os.path.exists(orig), f'{orig} is not a valid path!' 
    nlines = _count_examples(orig)

    strategy = strategies[args.strategy](args.eos_weights, 
                                         args.normalize, 
                                         args.weight_regularization)

    with open(dest, mode="w", encoding="utf8") as f:
        for instance in tqdm(read_tagged_file(orig), 
                             total=nlines,
                             desc='formating weights'):
            f.write(strategy.format_instance(instance) + '\n')