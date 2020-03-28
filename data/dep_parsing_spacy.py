"""
Uses Spacy to tokenize a dataset's *_output.txt file.
"""
import argparse
import os

import spacy
from spacy.tokens.doc import Doc
from tqdm import tqdm

tok_mapping = {
    '-lrb-': '(',
    '-rrb-': ')',
    '--': '-',
    '``': '"',
    "''": '"',
}


def read_sentence(f):
    if args.format == 'sent':
        pbar.update()
        return f.readline().strip().split()
    else:
        assert args.format == 'word'
        sentence = []
        while True:
            pbar.update()
            line = f.readline()
            if line.strip():
                token = line.split()[0]
                sentence.append(tok_mapping.get(token, token))
            else:
                return sentence


def main():
    print('Loading SpaCy parser...', end='')
    os.system('python3 -m spacy download en_core_web_lg > /dev/null')
    nlp = spacy.load("en_core_web_lg")
    print(' [ok]')

    with open(args.orig) as f_refs, open(args.dest, 'w') as f_tags:
        while pbar.n < pbar.total:
            sentence = Doc(nlp.vocab, read_sentence(f_refs))
            sentence = nlp.parser(sentence)
            for token in sentence:
                f_tags.write(f'{token.text}\t{token.dep_}\t{0 if token.dep_ == "root" else token.head.i}\n')
            f_tags.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # These arguments are for stand-alone file
    parser.add_argument('--orig', '-o', dest='orig',
                        help='Name of the stand alone file')
    parser.add_argument('--dest', '-d', dest='dest',
                        help='Name of the resulting file')
    parser.add_argument('--format', choices=['word', 'sent'], default='sent',
                        help='Format of the orig file:\n\t'
                             '"word" means one word per line (only the first one is considered)\n\t'
                             '"sent" means one sentence per line, and words will be identified using .split()')

    args = parser.parse_args()

    total = int(os.popen(f'wc -l < {args.orig}').read())
    pbar = tqdm(total=total)

    main()
