import argparse
from pprint import pprint

import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from scipy.stats import ttest_rel

from data.utils import FileIterable
from parent import parent_instance_level


def t_test(packed):
    stat, p_value = ttest_rel(*packed)
    return {'t-statistic': stat, 'p-value': p_value}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute instance-level BLEU and PARENT metrics for two different '
                                                 'hypothesis files. For each metric, perform a t-test over the two '
                                                 'files.')
    parser.add_argument('tables')
    parser.add_argument('references')
    parser.add_argument('hypotheses_1')
    parser.add_argument('hypotheses_2')

    args = parser.parse_args()

    tables = FileIterable.from_filename(args.tables, fmt='jl')
    references = FileIterable.from_filename(args.references, fmt='txt')
    hypotheses = [FileIterable.from_filename(args.hypotheses_1, fmt='txt'),
                  FileIterable.from_filename(args.hypotheses_2, fmt='txt')]

    bleus = tuple([sentence_bleu([r], h) for r, h in zip(references, hyps)] for hyps in hypotheses)
    parents = np.array([[parent_instance_level((h, r, t)) for h, r, t in zip(hyps, references, tables)]
                        for hyps in hypotheses])

    t_tests = {
        'bleu': t_test(bleus),
        **{f'parent_{mt}': t_test(parents[:, :, i]) for i, mt in enumerate(['p', 'r', 'F1'])}
    }

    pprint(t_tests)
