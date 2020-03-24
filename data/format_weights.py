import pkg_resources
import argparse
import os



def binary_format(orig):
    """
    We add a final score for </s> token
    """
    for example in read_tagged_file(orig):
        scores = [[1, 0] if score=='0' else [0, 1] for _, score in example] + [[1, 1]]
        yield ' '.join([f'{a}:{b}' for a, b in scores])

strategies = {
    'binary': binary_format
}


def read_tagged_file(path):
    sentences = list()
    sentence = list()
    with open(path, mode='r', encoding='utf8') as f:
        for line in f:
            if line.strip():
                sentence.append(line.strip().split())
            else:
                sentences.append(sentence)
                sentence = list()
    if sentence: sentences.append(sentence)  # deal with no empty last line
    return sentences


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

    format_func = strategies[args.strategy]
    with open(dest, mode="w", encoding="utf8") as f:
        for example in format_func(orig):
            f.write(example + '\n')