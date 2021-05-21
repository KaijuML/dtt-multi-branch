from utils import TaggedFileIterable

from overrides import overrides

import multiprocessing as mp

import pkg_resources
import argparse
import tqdm
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
                
        kwargs are there so that all strats are compatible
                
    In order to implement a new strategy, only the method `_score_weight` should be
    implemented. It should be a function (float --> List[float OR int])    
    """
    def __init__(self, eos_weights, normalize=False, weight_regularization=0, reverse=False, **kwargs):
        self.normalize = normalize
        self.weight_regularization = weight_regularization

        if eos_weights is None:
            raise ValueError('Please specify --eos_weights. These weights are '
                             'used for the End Of Sentence token. In the '
                             'original paper, these weights were set to [1, 0]'
                             ', meaning that the eos token is always the '
                             'responsability of the factuallness branch. Note '
                             'that if weight_regularization is set, it will '
                             'automatically be taken into account and you do '
                             'not need to include it in --eos_weights')
        assert all(isinstance(w, (float, int)) for w in eos_weights)
        if weight_regularization > 0:
            eos_weights.insert(0, weight_regularization)
        if normalize:
            eos_weights = [w / sum(eos_weights) for w in eos_weights]
        self.eos_weights = eos_weights
        self.reverse = reverse

    @staticmethod
    def chain_scores(scores):
        return ' '.join(':'.join(map(str, s)) for s in scores)

    def _score_weight(self, w):
        raise NotImplementedError

    def score_weight(self, w):
        s = self._score_weight(w)
        if self.reverse:
            s.reverse()
        if self.weight_regularization > 0:
            s.insert(0, self.weight_regularization)
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
        return [1, 0] if w==0 else [0, 1]
    

class SoftBinaryStrategy(Strategy):
    @overrides
    def _score_weight(self, w):
        """Soft version of BinaryStrategy."""
        return [1 - w, w]


class ThresholdsStrategy(Strategy):
    """
    Activate a branch depending on the importance of the weights
    For instance, with thresholds [0, 0.5] activate:
        1st branch when 0
        2nd when in ]0, 0.5]
        3rd when in ]0.5, 1]
        
    Binary is a special case for thresholds = [0]
    """
    
    @overrides
    def __init__(self, eos_weights, normalize, weight_regularization, thresholds, reverse, **kwargs):
        super().__init__(eos_weights, normalize, weight_regularization, reverse)
        self.thresholds = thresholds
        
    @overrides
    def _score_weight(self, w):
        ret = [0] * (len(self.thresholds) + 1)
        for idx, t in enumerate(self.thresholds):
            if w <= t:
                ret[idx] = 1
                return ret
        ret[-1] = 1
        return ret


class ContinousThresholdStrategy(ThresholdsStrategy):
    @overrides
    def _score_weight(self, w):
        n_ones = sum(w > t for t in self.thresholds) + 1
        return [1 if k < n_ones else 0 for k in range(len(self.thresholds)+1)]


class OneBranchStrategy(Strategy):
    @overrides
    def __init__(self, eos_weights, normalize, weight_regularization, reverse, **kwargs):
        super().__init__(eos_weights, normalize, weight_regularization, reverse)
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
    'soft_binary': SoftBinaryStrategy,
    'one_branch': OneBranchStrategy,
    'thresholds': ThresholdsStrategy,
    'continous': ContinousThresholdStrategy
}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    group = parser.add_argument_group('Original file and destination path')
    group.add_argument('--orig', '-o', dest='orig',
                        help='Name of the weights-tagged examples')
    group.add_argument('--dest', '-d', dest='dest',
                        help='Name of the resulting file')
    
    group = parser.add_argument_group('Arguments to create convert strategy')
    group.add_argument('--strategy', '-s', dest='strategy',
                        choices=list(strategies), default="binary",
                        help='Strategy to format weights.')
    group.add_argument('--normalize', dest='normalize', action='store_true',
                        help='Divide individual weights by the total sum.')
    group.add_argument('--weight_regularization', dest='weight_regularization',
                        type=float, default=0,
                        help='Weights for regularization branch. '
                             'Zero means no branch.')
    group.add_argument('--eos_weights', dest='eos_weights', nargs='+',
                        type=float, help='Weights to predict </s> token.')
    group.add_argument('--thresholds', dest='thresholds', nargs='+',
                        type=float, help='thresholds for ThresholdsStrategy.')
    group.add_argument('--reverse', dest='reverse', action='store_true',
                       help='reverse weights\' order')
    
    group = parser.add_argument_group('Arguments regarding multiprocessing')
    group.add_argument('--n_jobs', dest='n_jobs', type=int, default=-1,
                        help='number of processes to use. <0 for cpu_count()')
    group.add_argument('--chunksize', dest='chunksize', type=int, default=10,
                        help='chunksize to use in mp.Pool().imap() ' \
                             'Change this if you know what you are doing.')

    args = parser.parse_args()
    
    if not args.chunksize > 0:
        print('\nWARNING:',
              'Expected chunksize to be a non-zero positive integer.',
              f'Instead got {args.chunksize}.',
              'Instead, chunksize=1 will be used')
        args.chunksize = 1
        
    if os.path.exists(args.dest):
        print('\nWARNING:',
              f'destintation file {os.path.abspath(args.dest)} already exists, it will be overwritten.',
              'Stop the process ASAP to avoid this\n')
    else:
        # we use this touch to verify dest is a valid path
        # so that the script does not run if it's not the case
        print(f'Writting formatted wieghts to: {os.path.abspath(args.dest)}')
        with open(args.dest, mode="w", encoding='utf8') as f:
            pass 
    
    strategy = strategies[args.strategy](
        eos_weights=args.eos_weights, 
        normalize=args.normalize, 
        weight_regularization=args.weight_regularization,
        reverse=args.reverse,
        thresholds=args.thresholds
    )
    
    print('Reading orig file. Can take up to a minute.')
    scored_sentences = [
        sent for sent in TaggedFileIterable.from_filename(
            args.orig, func=lambda x,y: (x, float(y)))
    ]
    
    n_jobs = mp.cpu_count() if args.n_jobs < 0 else args.n_jobs
    print(f'Formatting weights, using {n_jobs} processes, starting now')
    with open(args.dest, mode="w", encoding='utf8') as f, mp.Pool(processes=n_jobs) as pool:
        _iterable = pool.imap(
            strategy.format_instance, 
            scored_sentences,
            chunksize=args.chunksize
        )
        
        for weights in tqdm.tqdm(
            _iterable, total=len(scored_sentences), desc='Formating weights'):
            f.write(weights + '\n')
