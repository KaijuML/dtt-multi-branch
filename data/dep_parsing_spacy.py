"""
Uses Spacy to tokenize a dataset's *_output.txt file.
"""

import os
from os import path

import spacy
from spacy.tokens.doc import Doc
from spacy.tokens.token import Token
from tqdm import tqdm

from co_occurrence import data_folder, num_examples

subset = 'train'
assert subset in ['train', 'valid', 'test']


def main():
    print('Loading SpaCy tokenizer...', end='')
    os.system('python3 -m spacy download en_core_web_lg > /dev/null')
    nlp = spacy.load("en_core_web_lg")
    print(' [ok]')

    with open(path.join(data_folder, f'{subset}_output.txt')) as f_refs, \
            open(path.join(data_folder, f'{subset}_deprel.txt'), 'w') as f_tags:
        for sentence in tqdm(f_refs, total=num_examples):
            sentence = Doc(nlp.vocab, sentence.split())
            sentence = nlp.tagger(sentence)
            for token in sentence:
                token: Token
                f_tags.write(f'{token.text}\t{token.dep_}\t{0 if token.dep_ == "root" else token.head.i}\n')
            f_tags.write('\n')


if __name__ == '__main__':
    main()
