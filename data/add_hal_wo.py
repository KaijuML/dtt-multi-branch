"""
Compute word overlap between sources and targets
"""

from utils import FileIterable
import numpy as np

import argparse
import tqdm
import json
import os


def word_overlap(table, sentence):
    x = {w for key, values in table for w in [key]+values}
    y = set(sentence)
    return 1 - (len(x.intersection(y)) / len(y))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    group = parser.add_argument_group('File system')
    group.add_argument('--orig_tbls', dest='orig_tbls',
                        help='Name of the tables file')
    group.add_argument('--orig_refs', dest='orig_refs',
                        help='Name of the references file')
    group.add_argument('--dest_prefix', dest='dest_prefix',
                        help='Prefix to add to modified files.')
    
        
    group = parser.add_argument_group('Hyperparameters')
    group.add_argument('--n_bins', dest='n_bins', type=int, default=5,
                       help='How many splits of overlap scores')
    
    
    args = parser.parse_args()
    
    tables = FileIterable.from_filename(args.orig_tbls, fmt='jl')
    outputs = FileIterable.from_filename(args.orig_refs)
    
    new_table_filename, _ = os.path.splitext(args.orig_tbls)
    new_table_filename = f'{new_table_filename}_{args.dest_prefix}.jl'
    
    n_bins = args.n_bins
    bins = np.arange(0+1/n_bins, 1.+1/n_bins, 1/n_bins)
    
    with open(new_table_filename, mode="w", encoding='utf8') as f:
        desc = "Adding Word Overlap Hallucination score"
        for table, output in tqdm.tqdm(zip(tables, outputs), desc=desc):
            _float_score = word_overlap(table, output)
            for bin, bin_limit in enumerate(bins):
                if _float_score <= bin_limit:
                    break
                    
            # adding tokens that we are sure are not in the vocabulary
            table.append(['HALWO', [f'[HALWO={bin}]']])
            
            f.write(json.dumps(table) + '\n')