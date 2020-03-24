"""
Creates the hallucination score file {subset}_h.txt, using information from PoS tags (assumed in {subset}_pos.txt)
and Spacy detected in-sentence dependencies.
"""

import json
import os
from collections import Counter
from os import path

import spacy
from spacy.tokens.doc import Doc
from tqdm import tqdm, trange

data_folder = path.join('data', 'wikibio')

subset = 'train'
assert subset in ['train', 'valid', 'test']

tables_filename = path.join(data_folder, f'{subset}_tables.jl')
pos_filename = path.join(data_folder, f'{subset}_pos.txt')
hallucination_filename = path.join(data_folder, f'{subset}_h.txt')

num_examples = int(os.popen(f'wc -l < {path.join(data_folder, f"{subset}_tables.jl")}').read())
num_words = int(os.popen(f'wc -l < {path.join(data_folder, f"{subset}_pos.txt")}').read())

interesting_tags = ['NOUN', 'ADJ', 'NUM', 'PROPN']

nlp = None


def ascend_dependency_tree(token):
    while token.dep_ not in ['appos', 'relcl', 'cc', 'prep', 'advcl', 'acl', 'amod', 'conj', 'compound']:
        if token.dep_ == 'ROOT':
            return None
        token = token.head
    return token


def build_sentence_object(token_list):
    global nlp
    if nlp is None:
        nlp = spacy.load("en_core_web_lg")

    # __init__
    sentence = Doc(nlp.vocab, [token for token, tag in token_list])
    # PoS tag
    for token in sentence:
        token.pos_ = token_list[token.i][1]
    sentence.is_tagged = True
    # Parse dependencies
    sentence = nlp.parser(sentence)
    return sentence


def expand(token, h):
    """
    Expands an hallucinated token
    i.e. tags the related (according to the dependence tree) tokens as hallucinations too
    """
    ancestor = ascend_dependency_tree(token)
    if ancestor:
        for t in ancestor.subtree:
            if t.pos_ in interesting_tags:
                h[t.i] = h[t.i] * h[token.i]
            else:
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
    for row in table:
        table_key = row[0]
        for table_value in row[1]:
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
        if (token.is_bracket or token.is_quote) and token.is_left_punct:
            j = 0
            min_h = 1
            for i_par, token_par in enumerate(sentence[token.i + 1:], start=token.i + 1):
                if token_par.is_right_punct and (
                        (token.is_bracket and token_par.is_bracket) or (token.is_quote and token_par.is_quote)
                ):
                    j = i_par
                    break
                else:
                    min_h = min(min_h, h[i_par])
            if j:
                h[token.i] = min_h
                h[j] = min_h


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


def main():
    #
    # STEP 1
    # Create co_occur Counter object
    #
    co_occur = dict()
    references = []
    with open(tables_filename) as tables_file, open(pos_filename) as refs_file:
        for _ in trange(num_examples, desc='Counting co-occurrences'):
            sentence, sent_interesting_tokens = read_sentence(refs_file)
            references.append(sentence)

            table = json.loads(tables_file.readline())
            for key_val in table:
                # [[<key_0>, [<token_00>, ..., <token_0m>]], ..., [<key_n>, [<token_n0>, ..., <token_nm>]]]
                table_items = [(key_val[0], val) for val in key_val[1]]

                for table_item in table_items:
                    # Filter out tokens that are equal to the table value (they are certainly not hallucinations)
                    tokens = [t for t in sent_interesting_tokens if t != table_item[-1]]
                    if table_item in co_occur:
                        co_occur[table_item].update(tokens)
                    else:
                        co_occur[table_item] = Counter(tokens)

    #
    # STEP 2
    # Keep only elements whose total co_occurrences number is in the 5th percentile
    # Convert counts to scores
    #
    keys = list(co_occur.keys())
    keys.sort(key=lambda key: max(co_occur[key].values()) if len(co_occur[key]) else 0, reverse=True)
    for i, k in tqdm(enumerate(keys), desc='Extracting common co-occurrences', total=len(keys)):
        if i / len(keys) < 0.05:
            max_cnt = max(co_occur[k].values())
            co_occur[k] = {t: hallucination_inv_score(c, max_cnt) for t, c in co_occur[k].items() if c > 5}
        else:
            del co_occur[k]

    #
    # STEP 3
    # Score each sentence token, according to the corresponding table and its co-occurrences
    #
    with open(tables_filename) as tables_file, open(hallucination_filename, 'w') as hallucination_file:
        for i, table in tqdm(enumerate(tables_file), desc='Scoring tokens', total=num_examples):
            scores = build_scores(table, co_occur)
            sentence = build_sentence_object(references[i])
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
    main()
