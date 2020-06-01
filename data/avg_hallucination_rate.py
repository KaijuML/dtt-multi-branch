from utils import TaggedFileIterable
from statistics import mean

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='<model>_h.txt file, containing the generated sentences, one token per line, '
                                     'each one along with its hallucination score. '
                                     'Sentences are separated by an empty line.')
    args = parser.parse_args()
    threshold = 0.4

    tagged_hypotheses = TaggedFileIterable.from_filename(args.file, func=lambda x, y: float(y))
    avg_score = mean([mean(h > threshold for h in sent_h) for sent_h in tagged_hypotheses])
    print(f'{args.file:70s}:\t{100 * avg_score:.2f} %')
