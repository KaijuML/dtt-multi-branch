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


def binary_format(example):
    """
    Tokens are either hallucinated or they're not.
    We add a final score for </s> token
    """
    scores = [[1, 0] if score=='0' else [0, 1] for _, score in example] + [[1, 1]]
    return ' '.join([f'{a}:{b}' for a, b in scores])


strategies = {
    'binary': binary_format
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
    
    args = parser.parse_args()

    folder = pkg_resources.resource_filename(__name__, 'wikibio')
    orig = os.path.join(folder, args.orig)
    dest = os.path.join(folder, args.dest)
    
    assert os.path.exists(orig), f'{orig} is not a valid path!' 
    nlines = _count_examples(orig)

    format_func = strategies[args.strategy]
    with open(dest, mode="w", encoding="utf8") as f:
        for example in tqdm(read_tagged_file(orig), 
                            total=nlines,
                            desc='formating weights:'):
            f.write(format_func(example) + '\n')