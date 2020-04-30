"""
Creates the hallucination score file {subset}_h.txt, using information from PoS tags (assumed in {subset}_pos.txt)
and Spacy detected in-sentence dependencies.
"""
from utils import FileIterable, TaggedFileIterable

from collections import Counter

import multiprocessing as mp

import itertools
import argparse
import tqdm
import json
import os
import math

interesting_tags = ['NOUN', 'ADJ', 'NUM', 'PROPN']


class Token:
    def __init__(self, i, text) -> None:
        super().__init__()
        self.i = i
        self.text = text
        self.children = set()
        self.pos_ = None
        self.dep_ = None
        self.head = None

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == 'head' and value:
            value.children.add(self)

    @property
    def subtree(self):
        s = {self}
        for c in self.children:
            s.update(c.subtree)
        return s

    def __repr__(self):
        return self.text


def ascend_dependency_tree(token):
    while token.dep_ not in ['acl', 'advcl', 'amod', 'appos', 'cc', 'compound', 'conj', 'discourse', 'infmod', 'iobj',
                             'list', 'nmod', 'obl', 'orphan', 'partmod', 'prep', 'rcmod', 'relcl']:
        if token.dep_.upper() == 'ROOT':
            return None
        token = token.head
    return token


def build_sentence_object(token_list):
    # __init__
    sentence = [Token(i, t[0]) for i, t in enumerate(token_list)]
    # Read PoS tags and dependency parsing information
    for token in sentence:
        token.pos_ = token_list[token.i][1]
        token.dep_ = token_list[token.i][2]
        token.head = None if token.dep_.upper() == 'ROOT' else sentence[token_list[token.i][3]]
    return sentence


def expand(token, h):
    """
    Expands an hallucinated token
    i.e. tags the related (according to the dependence tree) tokens as hallucinations too
    """
    ancestor = ascend_dependency_tree(token)
    if ancestor:
        for t in ancestor.subtree:
            if t.pos_ not in interesting_tags:
                h[t.i] = max(h[t.i], h[token.i])


def build_scores(table, co_occur):
    """

    Args:
        table: The input table as list([str, list(str)])
        co_occur: The general co_occurrence score dictionary

    Returns: The table's token hallucination inverse score dictionary

    """
    
    # flatten the table --> {<key_0>, <token_00>, ..., <token_0m>, <key_1>, ..., <token_nm>}
    tokens_in_table = set(itertools.chain(*[[key] + val for key, val in table]))

    # scores assigns 1 to all the table keys and values
    scores = {t: 1 for t in tokens_in_table}

    # scores assigns a proper value to the table rows' co-occurrences
    for key, values in table:
        for val in values:
            table_tuple = key, val
            for token, score in co_occur.get(table_tuple, dict()).items():
                scores[token] = max(score, scores.get(token, 0))
    return scores


def hallucination_inv_score(cnt, max_cnt):
    a = 1 / (max_cnt - 5) ** 2
    return min(max(a * cnt ** 2 - 10 * a * cnt + 25 * a, 0), 1)


def handle_sentence_punctuation(sentence, h):
    """
    Adjusts the hallucination scores using simple punctuation-related heuristics
    """
    for token in sentence:
        # Conjunctions and comma near hallucinations are hallucinated
        if token.i > 0 and (token.dep_ == 'cc' or token.text == ','):
            h[token.i] = max(h[token.i - 1:token.i + 2])
            # ", and" is hallucinated as a whole
            if token.dep_ == 'cc' and sentence[token.i - 1].text == ',':
                h[token.i - 1] = h[token.i]
        # Brackets/quotes containing hallucinations only are hallucinated
        ignore = set()
        if token.text in ['(', '"'] and token not in ignore:
            j = 0
            min_h = 1
            for i_par, token_par in enumerate(sentence[token.i + 1:], start=token.i + 1):
                if (token_par.text == ')' and token.text == '(') or (token_par.text == '"' and token.text == '"'):
                    j = i_par
                    ignore.add(token_par)
                    break
                else:
                    min_h = min(min_h, h[i_par])
            if j:
                h[token.i] = min_h
                h[j] = min_h


def fuse_pos_and_deprel(sentence_pos, sentence_dep=None):
    sentence = list()
    if sentence_dep is None:
        sentence_dep = [[None, None, None] for _ in range(len(sentence_pos))]

    assert len(sentence_pos) == len(sentence_dep), sentence_pos + ['//'] + sentence_dep
        
    for (word, tag), (_word, rel, head) in zip(sentence_pos, sentence_dep):
        
        # sanity check
        assert _word is None or _word == word, f'<{_word}> vs <{word}>'
        
        # noinspection PyTypeChecker
        sentence.append((word, tag) + ((rel.split(':')[0], int(head)) if _word is not None else ()))
        
    return sentence


def count_co_occurrences(filename, tables_loc, pos_loc):
    """
    Count co-occurrences in the training set. 
    """
    
    # If filename exists, simply returned cached co-occurrences
    if os.path.exists(filename):
        print(f'Loading co-occurrences counts from {filename}')
        with open(filename, mode="r", encoding='utf8') as f:
            return {
                tuple(key.split()): value
                for key, value in json.load(f).items()
            }

    print('Counting co-occurrences between source tables and target sentences')
        
    tables = FileIterable.from_filename(tables_loc, fmt='jl')
    sentences = TaggedFileIterable.from_filename(pos_loc)
    
    # Creating co_occur dict
    co_occur = dict()

    for table, sentence in tqdm.tqdm(zip(tables, sentences), 
                         desc='Counting co-occurrences', 
                         total=len(tables)):

        for key, values in table:
            table_items = [(key, value) for value in values]
            for table_item in table_items:
                # Only include interestingly tagged tokens that are not equal to the table value
                # noinspection PyTypeChecker
                co_occur.setdefault(table_item, Counter()) \
                    .update([token for token, pos in sentence
                             if pos in interesting_tags and token != table_item[-1]])
    
    # We filter the co_occur dictionary.
    # We keep only elements whose total co_occurrences number is in the 5th percentile
    # We also convert raw counts to normalized scores in [0, 1]
    
    print('Sorting keys')
    sorted_keys = sorted(
        co_occur, 
        key=lambda x: max(co_occur[x].values()) if len(co_occur[x]) else 0,
        reverse=True
    )[1:]  # I remove the first key because I have manually seen that it's useless
    print('Key are sorted')
    
    co_occur = {
        key: {
            t: hallucination_inv_score(c, max(co_occur[key].values()))
            for t, c in co_occur[key].items()
            if c > 5
        }
        # we iterate through all keys
        for idx, key in tqdm.tqdm(enumerate(sorted_keys), 
                                  desc='Extracting common co-occurrences',
                                  total=len(sorted_keys))
        
        # we only keep the top 5th percentile
        if idx / len(sorted_keys) < 0.05
    }

    # We cache the resulting dict to save time later
    print(f'Serializing co-occurrences dict to {filename}')
    with open(filename, mode='w', encoding='utf8') as f:
        json.dump({' '.join(key): val for key, val in co_occur.items()}, f)

    return co_occur


tf: dict
idf: dict
co_occur: dict

def _tf_idf(r):
    return r, {t: tf[r][t] * idf[t] for t in idf.keys()}


def _tf(keyval):
    r, cnt = keyval
    return r, {t: co_occur[r][t] / sum(co_occur[r].values()) for t in cnt.keys()}


def _idf(t):
    return t, math.log2(len(co_occur) / len(list(filter(lambda r: t in co_occur[r].keys(), co_occur))))


def compute_tf_idf(filename, tables_loc, pos_loc):
    """
    Count co-occurrences in the training set.
    """

    # If filename exists, simply returned cached co-occurrences
    if os.path.exists(filename):
        print(f'Loading tf-idf from {filename}')
        with open(filename, mode="r", encoding='utf8') as f:
            return {
                tuple(key.split()): value
                for key, value in json.load(f).items()
            }

    print('Counting co-occurrences between source tables and target sentences')

    tables = FileIterable.from_filename(tables_loc, fmt='jl')
    sentences = TaggedFileIterable.from_filename(pos_loc)

    # Creating co_occur dict
    global co_occur
    co_occur = dict()

    for table, sentence in tqdm.tqdm(zip(tables, sentences),
                                     desc='Counting co-occurrences',
                                     total=len(tables)):

        for key, values in table:
            table_items = [(key, value) for value in values]
            for table_item in table_items:
                # Only include interestingly tagged tokens that are not equal to the table value
                # noinspection PyTypeChecker
                co_occur.setdefault(table_item, Counter()) \
                    .update([token for token, pos in sentence
                             if pos in interesting_tags and token != table_item[-1]])

    all_tokens = set.union(*[set(cnt.keys()) for cnt in co_occur.values()])

    global tf
    global idf
    n_jobs = mp.cpu_count() if args.n_jobs < 0 else args.n_jobs
    with mp.Pool(n_jobs) as pool:
        tf = pool.imap_unordered(
            _tf,
            tqdm.tqdm(co_occur.items(), desc='TF'),
            chunksize=args.chunksize)
        tf = dict(tf)

        idf = pool.imap_unordered(
            _idf,
            tqdm.tqdm(all_tokens, desc='IDF'),
            chunksize=args.chunksize)
        idf = dict(idf)

        tf_idf = pool.imap_unordered(_tf_idf,
                                     tqdm.tqdm(tf.keys(), desc='TF-IDF'),
                                     chunksize=args.chunksize)
        tf_idf = dict(tf_idf)

    breakpoint()
    del tf, idf, co_occur

    # We cache the resulting dict to save time later
    print(f'Serializing tf-idf dict to {filename}')
    with open(filename, mode='w', encoding='utf8') as f:
        json.dump({' '.join(key): val for key, val in tf_idf.items()}, f)

    # TODO sort?
    # TODO filter rows?
    # TODO filter row tokens?
    # TODO convert score?

    return tf_idf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    group = parser.add_argument_group('Arguments to parametrize the co-occurrences count')
    group.add_argument('--frequencies', '-f', required=True,
                        help='Cached file containing filtered co-occurrences dict. '
                             'If it does not exists, it will be used to store the co-occurrences count')
    group.add_argument('--freq-input', '-fin', help='The .jl tables file used to compute co-occurrences')
    group.add_argument('--freq-pos', '-fpos', help='The references PoS file used to compute co-occurrences')

    
    group = parser.add_argument_group('Arguments to parametrize the scoring')
    group.add_argument('--input', '-i', required=True,
                        help='The input tables file')
    group.add_argument('--pos', '-p', required=True,
                        help='PoS outputs file')
    group.add_argument('--deprel', '-d', required=True,
                        help='Dependency relations outputs file')
    group.add_argument('--scores', '-s', required=True,
                       help='Scores file to write')
    group.add_argument('--score_type', '-t', choices=['quad', 'tf-idf'], default='quad',
                       help='How to calculate the scores from co-occurrences: \n'
                            '\tquad:   quadratic decay from max count (scored 1) to 5 (scored 0)'
                            '\ttf-idf: term frequency–inverse document frequency (TF-IDF)')

    group = parser.add_argument_group('Arguments regarding multiprocessing')
    group.add_argument('--n_jobs', dest='n_jobs', type=int, default=-1,
                        help='number of processes to use. <0 for cpu_count()')
    group.add_argument('--chunksize', dest='chunksize', type=int, default=10,
                        help='chunksize to use in mp.Pool().imap()' \
                             'Change this if you know what you are doing.')

    args = parser.parse_args()
    
    if not args.chunksize > 0:
        print('\nWARNING:',
              'Expected chunksize to be a non-zero positive integer.',
              f'Instead got {args.chunksize}.',
              'Instead, chunksize=1 will be used')
        args.chunksize = 1
    
    if os.path.exists(args.scores):
        print('\nWARNING:',
              f'{args.scores} already exists, it will be overwritten.',
              'Stop the process ASAP to avoid this\n')
    else:
        # we use this touch to verify dest is a valid path
        # so that the script does not run if it's not the case
        with open(args.scores, mode="w", encoding='utf8') as f:
            pass 

    # Count co-occurences in the training set
    co_occur = {
        'quad': count_co_occurrences,
        'tf-idf': compute_tf_idf
    }[args.score_type](args.frequencies, args.freq_input, args.freq_pos)

    print('loading source tables, PoS-tagging and DependencyRelations-tagging')
    tables = FileIterable.from_filename(args.input, fmt='jl')
    sentences_pos = TaggedFileIterable.from_filename(args.pos)
    sentences_dep = TaggedFileIterable.from_filename(args.deprel)
    
    zipped_inputs = [
        item for item in tqdm.tqdm(
            zip(tables, sentences_pos, sentences_dep),
            desc='Reading files',
            total=len(tables)
        )
    ]
    
    def deal_with_one_instance(zipped_args):
    
        input_table, input_pos, input_deprel = zipped_args 

        scores = build_scores(input_table, co_occur)
        sentence = build_sentence_object(fuse_pos_and_deprel(input_pos, input_deprel))

        h = [int(token.pos_ in interesting_tags) for token in sentence]

        # Score each token
        for token in sentence:
            if h[token.i]:
                h[token.i] -= scores.get(token.text, 0)

        # expand the hallucination scores
        for token in sentence:
            if token.pos_ in interesting_tags and h[token.i] > 0:
                expand(token, h)

        # Some tricks to harmonize the scoring
        handle_sentence_punctuation(sentence, h)

        return sentence, h
    
    n_jobs = mp.cpu_count() if args.n_jobs < 0 else args.n_jobs
    print(f'Using {n_jobs} processes, starting now')
    with open(args.scores, mode="w", encoding='utf8') as f, mp.Pool(processes=n_jobs) as pool:
        _iterable = pool.imap(
            deal_with_one_instance, 
            zipped_inputs,
            chunksize=args.chunksize
        )
        
        for sentence, scores in tqdm.tqdm(
            _iterable, total=len(tables), desc='Scoring references'):
            
            for token in sentence:
                f.write(f'{token.text}\t{scores[token.i]}\n')
            f.write('\n')
