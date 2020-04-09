"""
This scripts filters the references from WikiBIO using our custom token score function
For now, only token with a score > 0 are kept.
"""


import multiprocessing as mp

import more_itertools
import overrides
import argparse
import tqdm
import os


class FileIterable:
    def __init__(self, iterable):
        self._iterable = more_itertools.seekable(iterable)
        self._ptr = 0  # pointer to the next item
        self._len = None
        
    @classmethod
    def from_file(cls, path, func=None):
        return cls(cls.read_file(path, func))
    
    @staticmethod
    def read_file(path, func=None):
        """
        ARGS:
            path (str): hopefully a valid path. 
                        Each line contains a sentence
                        Tokens are separated by a space ' '
            func (NoneType): Should be None in this simple case
        """
        assert func is None
        with open(path, mode="r", encoding="utf8") as f:
            for line in f:
                yield line.strip().split()
    
    def __getitem__(self, n):
        self._iterable.seek(n)
        self._ptr = n
        try:
            return next(self)
        except StopIteration:
            raise IndexError(f'{n} is outside this generator')
            
        return ret
    
    def __next__(self):
        self._ptr += 1
        return next(self._iterable)
    
    def __len__(self):
        if self._len is None:
            ptr = self._ptr  # remember the position of the pointer
            length = more_itertools.ilen(self._iterable)  # count remaining items
            _ = self[ptr - 1]  # set back the pointer
            self._len = length + ptr
        return self._len


class TaggedFileIterable(FileIterable):
    
    @staticmethod
    @overrides.overrides
    def read_file(path, func=None):
        """
        Accumulate lines until empty line, then yield and do it again
        ARGS:
            path (str): hopefully a valid path. 
                        Each line contains a token and its tag(s)
                        Sentences are separated by an empty line
            func (func): will be applied to each line. 
                         Defaults to identity func `lambda x: x`
                         Use this to convert tags to correct type
        """
        if func is None:
            def func(*args):
                return args
        
        sentence = list()
        with open(path, mode='r', encoding='utf8') as f:
            for line in f:
                if line.strip():
                    sentence.append(func(*line.strip().split()))
                else:
                    yield sentence
                    sentence = list()
        if sentence: 
            yield sentence


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    group = parser.add_argument_group('How to config paths')
    group.add_argument('--dest', dest='dest', required=True,
                       help='Where to store the filtered references')
    group.add_argument('--scores', dest='scores', required=True,
                       help='Each line is (token, score) separated by \\t ' \
                            'Sentences are separated by an empty line')
    group.add_argument('--refs', dest='refs', required=True,
                       help='Reference file. One sentence per line.')
    group.add_argument('--threshold', dest='threshold', type=float, default=0,
                       help='Only keep tokens with a score <= threshold')
    
    
    group = parser.add_argument_group('Arguments regarding multiprocessing')
    group.add_argument('--n_jobs', dest='n_jobs', type=int, default=-1,
                        help='number of processes to use. <0 for cpu_count()')
    group.add_argument('--chunksize', dest='chunksize', type=int, default=10,
                        help='chunksize to use in mp.Pool().imap()' \
                             'Change this if you know what you are doing.')
    
    args = parser.parse_args()
    
    if not 0 <= args.threshold <= 1:
        raise ValueError('threshold should be between 0 and 1'
                         f'Got {args.threshold}')
        
    if not args.chunksize > 0:
        print('\nWARNING:',
              'Expected chunksize to be a non-zero positive integer.',
              f'Instead got {args.chunksize}.',
              'Instead, chunksize=1 will be used')
        args.chunksize = 1
    
    if os.path.exists(args.dest):
        print('\nWARNING:',
              f'{args.dest} already exists, it will be overwritten.',
              'Stop the process ASAP to avoid this\n')
    else:
        # we use this touch to verify dest is a valid path
        # so that the script does not run if it's not the case
        with open(args.dest, mode="w", encoding='utf8') as f:
            pass
        
    
    references = FileIterable.from_file(args.refs)
    scored_references = TaggedFileIterable.from_file(args.scores, 
                                                     func=lambda x, s: (x, float(s)))
    
    zipped_inputs = [
        item for item in tqdm.tqdm(
            zip(references, scored_references),
            desc='Reading files',
            total=len(references)
        )
    ]
    
    def deal_with_one_instance(zipped_args):
        ref, scored_ref = zipped_args
        filtered_ref = list()
        for token, (_, score) in zip(ref, scored_ref):
            if score <= args.threshold:
                filtered_ref.append(token)
        return ' '.join(filtered_ref)
    
    n_jobs = mp.cpu_count() if args.n_jobs < 0 else args.n_jobs
    print(f'Using {n_jobs} processes, starting now')
    with open(args.dest, mode="w", encoding='utf8') as f, mp.Pool(processes=n_jobs) as pool:
        _iterable = pool.imap(
            deal_with_one_instance, 
            zipped_inputs,
            chunksize=args.chunksize
        )
        
        for filtered_reference in tqdm.tqdm(
            _iterable, total=len(references), desc='Filtering references'):
            f.write(f'{filtered_reference}\n')