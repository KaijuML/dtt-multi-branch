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

num_references = int(os.popen(f'wc -l < {path.join(data_folder, f"{subset}_tables.jl")}').read())
num_words = int(os.popen(f'wc -l < {path.join(data_folder, f"{subset}_pos.txt")}').read())

interesting_tags = ['NOUN', 'ADJ', 'NUM', 'PROPN']
min_cnt = 100
most_common_cnt = 10

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


def counter_to_perc(c):
    """Converts a Counter object to a general dict whose values are expressed as percent of the total original count."""
    total = sum(c.values())
    return {k: v / total for k, v in c.items()}


def dict_to_tuples(d, prefix=tuple()):
    """
    Extract tuples representing nested tables elements.
    Needed for more sophisticated datasets such as Rotowire/SBNation.

    Example:
        >>> d = {'a': 0, 'b': {'c': 1, 'd': 2}}
        >>> dict_to_tuples(d)
        [('a', 0), ('b', 'c', 1), ('b', 'd', 2)]
    """
    tuples = []
    for key, value in d.items():
        if type(value) is dict:
            tuples += dict_to_tuples(value, prefix=prefix + (key,))
        else:
            tuples.append(prefix + (key, value))
    return tuples


def expand(token, h):
    ancestor = ascend_dependency_tree(token)
    if ancestor:
        for t in ancestor.subtree:
            if not h[t.i]:
                h[t.i] += 1


def get_allowed_words(table, co_occur):
    # [[<key_0>, [<token_00>, ..., <token_0m>]], ..., [<key_n>, [<token_n0>, ..., <token_nm>]]]
    table = json.loads(table)
    # {<token_00>, ..., <token_nm>}
    allowed_words = set(sum([[key_val[0]] + key_val[1] for key_val in table], []))

    # allowed_words now contains all the table keys and values

    # [(<key_0>, "<token_00> ... <token_0m>"), ..., (<key_n>, "<token_n0> ... <token_nm>")]
    keys = []
    for key_val in table:
        keys += [(key_val[0], val) for val in key_val[1]]

    allowed_words = allowed_words.union(*[co_occur[k] for k in keys])  # Set of all table values + most f
    # allowed_words now contains all the table values + most common co-occurrences
    return allowed_words


def handle_sentence_punctuation(sentence, h):
    for token in sentence:
        # Conjunctions and comma near hallucinations are hallucinated
        if (token.dep_ == 'cc' or token.text == ',') \
                and not h[token.i] \
                and ((token.i != 0 and h[token.i - 1]) or (token.i != len(sentence) - 1 and h[token.i + 1])):
            h[token.i] += 1
            # ", and" is hallucinated as a whole
            if token.dep_ == 'cc' and sentence[token.i - 1].text == ',':
                h[token.i - 1] += 1
        # Brackets/quotes containing hallucinations only are hallucinated
        if (token.is_bracket or token.is_quote) and token.is_left_punct:
            j = 0
            for i_par, token_par in enumerate(sentence[token.i + 1:], start=token.i + 1):
                if (token.is_bracket and token_par.is_bracket and token_par.is_right_punct) or \
                        (token.is_quote and token_par.is_quote and token_par.is_right_punct):
                    j = i_par
                    break
                if not h[i_par]:
                    break
            if j:
                h[token.i] += 1
                h[j] += 1


def is_hallucinated(token, allowed_words):
    return token.pos_ in interesting_tags and not any(token.text in w for w in allowed_words)


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
        for _ in trange(num_references, desc='Counting co-occurrences'):
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
    # Extract the most common tokens for every table element
    # i.e. filter co_occur according to min_cnt and most_common_cnt
    #
    for k in tqdm(co_occur.keys(), desc='Extracting common co-occurrences'):
        if sum(co_occur[k].values()) > min_cnt:
            co_occur[k] = set(dict(co_occur[k].most_common(most_common_cnt)).keys())
        else:
            co_occur[k] = set()

    #
    # STEP 3
    # Score each sentence token, according to the corresponding table and its co-occurrences
    #
    with open(tables_filename) as tables_file, open(hallucination_filename, 'w') as hallucination_file:
        for i, table in tqdm(enumerate(tables_file)):
            allowed_words = get_allowed_words(table, co_occur)
            sentence = build_sentence_object(references[i])
            h = [0 for _ in range(len(sentence))]

            # Score each token and expand the hallucination score if necessary
            for token in sentence:
                if is_hallucinated(token, allowed_words):
                    h[token.i] += 2
                    expand(token, h)

            # Some tricks to harmonize the scoring
            handle_sentence_punctuation(sentence, h)

            # Write output to file {set}_h.txt
            for token in sentence:
                hallucination_file.write(f'{token.text}\t{h[token.i]}\n')
            hallucination_file.write('\n')


if __name__ == '__main__':
    main()
