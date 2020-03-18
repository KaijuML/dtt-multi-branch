"""
This scripts precompute stats for training more efficiently with RL on parent
Be aware that some liberties have been taken with the implementation so that 
everything runs smoothly and easily. It is not a duplicate of the original
PARENT metric. For evaluation, continue to use the original implementation!
"""

import itertools, collections, json
import argparse
import os


def nwise(iterable, n):
    iterables = itertools.tee(iterable, n)
    [next(iterables[i]) for i in range(n) for j in range(i)]
    return zip(*iterables)


def ngram_counts(sequence, order):
    """Returns count of all ngrams of given order in sequence."""
    if len(sequence) < order:
        return collections.Counter()
    return collections.Counter(nwise(sequence, order))


def overlap_probability(ngram, table_values):
    return len(table_values.intersection(ngram)) / len(ngram)



def load_tables(dataset, setname):
    tables_filename = os.path.join(dataset, f"{setname}_tables.jl")
    with open(tables_filename, encoding="utf8", mode="r") as tables_file:
        tables = [json.loads(line) for line in tables_file]
    return tables


def load_refs(dataset, setname):
    refs_filename =os.path.join(dataset, f"{setname}_output.txt")
    with open(refs_filename, encoding="utf8", mode="r") as refs_file:
        refs = [line.strip().split(" ")
                for line in refs_file if line.strip()]
    return refs


def serialize_stats(tv, rnc, rnw):
    tv = [list(s) for s in tv]
    rnc = [{order: {' '.join(ngram): count for ngram, count in counter.items()}
            for order, counter in rnc.items()}
           for rnc in rnc]
    rnw = [{order: {' '.join(ngram): count for ngram, count in counter.items()}
            for order, counter in rnw.items()}
           for rnw in rnw]
    return tv, rnc, rnw


def main(dataset):
    references = load_refs(dataset, setname='train')
    tables = load_tables(dataset, setname='train')

    if dataset == 'wikibio':
        TABLE_VALUES = [{tok for _, value in table for tok in value} for table in tables]
    else:
        TABLE_VALUES = [{tok for head, _, tail in table for tok in head + tail} for table in tables]

    REF_NGRAM_COUNTS = [{order: ngram_counts(ref, order)
                         for order in range(1, 5)} 
                        for ref in references]
    REF_NGRAM_WEIGHTS = [
        {
            order: {ngram: overlap_probability(ngram, table_values)
                    for ngram in counter}
            for order, counter in ref_counts_at_k.items()
        }
        for ref_counts_at_k, table_values in zip(REF_NGRAM_COUNTS, TABLE_VALUES)
    ]
            
    tv, rnc, rnw = serialize_stats(TABLE_VALUES,
                                   REF_NGRAM_COUNTS,
                                   REF_NGRAM_WEIGHTS)
            
    path = f'{dataset}/TABLE_VALUES.json'
    with open(path, encoding="utf8", mode="w") as f:
        json.dump(tv, f)
                
    path = f'{dataset}/REF_NGRAM_COUNTS.json'
    with open(path, encoding="utf8", mode="w") as f:
        json.dump(rnc, f)
                
    path = f'{dataset}/REF_NGRAM_WEIGHTS.json'
    with open(path, encoding="utf8", mode="w") as f:
        json.dump(rnw, f)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', dest='dataset', default="webnlg",
                       choices=['wikibio', 'webnlg'])
    
    args = parser.parse_args()
    
    main(args.dataset)
