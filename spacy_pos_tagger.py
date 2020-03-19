"""
Uses Spacy to tokenize a dataset's *_output.txt file.
"""

import os
from os import path

import spacy
from spacy.tokens.doc import Doc
from tqdm import tqdm

from co_occurrence import data_folder, num_references

subset = 'train'
assert subset in ['train', 'valid', 'test']


def main():
    print('Loading SpaCy tokenizer...', end='')
    os.system('python3 -m spacy download en_core_web_lg > /dev/null')
    nlp = spacy.load("en_core_web_lg")
    print(' [ok]')

    with open(path.join(data_folder, f'{subset}_output.txt')) as f_refs, \
            open(path.join(data_folder, f'{subset}_pos_spacy.txt'), 'w') as f_tags:
        for sentence in tqdm(f_refs, total=num_references):
            sentence = Doc(nlp.vocab, sentence.split())
            sentence = nlp.tagger(sentence)
            for token in sentence:
                f_tags.write(f'{token.text}\t{token.pos_}\n')
            f_tags.write('\n')


if __name__ == '__main__':
    main()
