"""
Creates the hallucination score file {subset}_h.txt, using information from PoS tags (assumed in {subset}_pos.txt)
and Spacy detected in-sentence dependencies.
"""
from collections import Counter
from tqdm import tqdm, trange
from os.path import exists

import argparse
import json
import os


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
                             'list', 'obj', 'obl', 'orphan', 'partmod', 'prep', 'rcmod', 'relcl']:
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
        token.head = None if token.dep_ == 'ROOT' else sentence[token_list[token.i][3]]
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
    # [[<key_0>, [<token_00>, ..., <token_0m>]], ..., [<key_n>, [<token_n0>, ..., <token_nm>]]]
    table = json.loads(table)
    # {<key_0>, <token_00>, ..., <token_0m>, <key_1>, ..., <token_nm>}
    tokens_in_table = set(sum([[key_val[0]] + key_val[1] for key_val in table], []))

    # scores assigns 1 to all the table keys and values
    scores = {t: 1 for t in tokens_in_table}

    # scores assigns a proper value to the table rows' co-occurrences
    for table_key, table_values in table:
        for table_value in table_values:
            table_tuple = (table_key, table_value)
            for token, score in co_occur.get(table_tuple, {}).items():
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


def read_sentence(pos_file, deprel_file=None):
    sentence = []
    while True:
        tagged_word = pos_file.readline()
        parsed_word = deprel_file.readline() if deprel_file else None
        if tagged_word.strip():
            word, tag = tagged_word.split()
            word_check, rel, head = parsed_word.split() if parsed_word else 3 * (None,)
            assert word_check is None or word_check == word
            # noinspection PyTypeChecker
            sentence.append((word, tag) + ((rel, int(head)) if deprel_file else ()))
        else:
            return sentence
        

def count_co_occurrences(filename, tables_loc, pos_loc):
    """
    Count co-occurrences in the training set. 
    """
    
    # If filename exists, simply returned cached co-occurrences
    if os.path.exists(filename):
        with open(filename, mode="r", encoding='utf8') as f:
            return {
                tuple(key.split()): value
                for key, value in json.load(f).items()
            }
        
    
    # Counting number of examples (by counting number of tables)
    num_examples = int(os.popen(f'wc -l < {tables_loc}').read())
    
    # Creating co_occur dict
    co_occur = dict()
    
    with open(tables_loc, mode="r", encoding='utf8') as tables_file, \
         open(pos_loc, mode="r", encoding="utf8") as refs_file:
        
        for _ in trange(num_examples, desc='Counting co-occurrences'):
            
            # reading the next example from both input/output files
            table = json.loads(tables_file.readline())  # source table
            sentence = read_sentence(refs_file)         # target sentence
            
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
    
    sorted_keys = sorted(
        co_occur, 
        key=lambda x: max(co_occur[x].values()) if len(co_occur[x]) else 0,
        reverse=True
    )
    
    co_occur = {
        key: {
            t: hallucination_inv_score(c, max(co_occur[key].values()))
            for t, c in co_occur[key].items()
            if c > 5
        }
        # we iterate through all keys
        for idx, key in tqdm(enumerate(sorted_keys), 
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

def main():
    
    # Count co-occurences in the training set
    co_occur = count_co_occurrences(args.frequencies, args.freq_input, args.freq_pos)
    
    return
    
    # Score each sentence token, according to the corresponding table and its co-occurrences
    with open(args.input) as tables_file, open(args.scores, 'w') as hallucination_file, \
            open(args.pos) as pos_file, open(args.deprel) as deprel_file:

        num_examples = int(os.popen(f'wc -l < {args.input}').read())
        for i, table in tqdm(enumerate(tables_file), desc='Scoring tokens', total=num_examples):
            scores = build_scores(table, co_occur)
            sentence = build_sentence_object(read_sentence(pos_file, deprel_file))
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

            # Write output to file {set}_h.txt
            for token in sentence:
                hallucination_file.write(f'{token.text}\t{h[token.i]}\n')
            hallucination_file.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frequencies', '-f', required=True,
                        help='Cached file containing filtered co-occurrences dict. '
                             'If it does not exists, it will be used to store the co-occurrences count')
    parser.add_argument('--freq-input', '-fin', help='The .jl tables file used to compute co-occurrences')
    parser.add_argument('--freq-pos', '-fpos', help='The references PoS file used to compute co-occurrences')

   # parser.add_argument('--input', '-i', required=True,
    #                    help='The input tables file')
    #parser.add_argument('--pos', '-p', required=True,
     #                   help='PoS outputs file')
    #parser.add_argument('--deprel', '-d', required=True,
     #                   help='Dependency relations outputs file')
    #parser.add_argument('--scores', '-s', required=True,
     #                   help='Scores file to write')

    args = parser.parse_args()

    main()
