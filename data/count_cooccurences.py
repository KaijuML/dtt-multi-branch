from collections import Counter
from tqdm import trange, tqdm

import pkg_resources
import argparse
import json
import os


interesting_tags = ['NOUN', 'ADJ', 'NUM', 'PROPN']


def read_sentence(refs_file):
    sentence = []
    sent_interesting_tokens = []
    while True:
        tagged_word = refs_file.readline()
        if tagged_word.strip():
            word, tag = tagged_word.split()
            if tag in interesting_tags:
                sent_interesting_tokens.append(word)
            sentence.append((word, tag))
        else:
            return sentence, sent_interesting_tokens


def main(folder, most_common_cnt, min_cnt):
    num_examples = int(os.popen(f'wc -l < {os.path.join(folder, f"train_tables.jl")}').read())
    
    co_occur = dict()
    
    tables_filename = os.path.join(folder, 'train_tables.jl')
    pos_filename = os.path.join(folder, 'train_pos.txt')
    with open(tables_filename) as tables_file, open(pos_filename) as refs_file:
        for _ in trange(num_examples, desc='Counting co-occurrences'):
            
            # reading files for next example
            sentence, sent_interesting_tokens = read_sentence(refs_file)
            table = json.loads(tables_file.readline())

            # counting all co-occurences in this examples
            for key, values in table:
                table_items = [(key, val) for val in values]
                for table_item in table_items:
                    co_occur.setdefault(table_item, Counter())
                    co_occur[table_item].update(t for t in sent_interesting_tokens if t != table_item[-1])

    co_occur = {
        ' '.join(key): dict(counter.most_common(most_common_cnt))
        for key, counter in tqdm(co_occur.items(), total=len(co_occur), desc='Filtering co-occurences')
        if sum(counter.values()) > min_cnt
    }
    
    cooc_filename = os.path.join(folder, 'occurences.json')
    with open(cooc_filename, mode="w", encoding='utf8') as f:
        json.dump(co_occur, f)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', dest='folder',
                        help='Name of the folder where train_pos.txt and train_tables.jl are.')
    parser.add_argument('--most_common_cnt', dest='most_common_cnt', type=int,
                        help='Number of tokens to keep for each table value')
    parser.add_argument('--min_cnt', dest='min_cnt', type=int,
                        help='Only keep a table value if it has at least min_cnt co-occurences')
    
    args = parser.parse_args()
    
    folder = pkg_resources.resource_filename(__name__, 'wikibio')
    folder = os.path.join(folder, args.folder)
    
    main(folder, args.most_common_cnt, args.min_cnt)