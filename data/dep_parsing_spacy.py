"""
Uses Spacy to tokenize a dataset's *_output.txt file.
"""
from spacy.tokens.doc import Doc

import multiprocessing as mp
import numpy as np

import argparse
import spacy
import tqdm
import time
import os


tok_mapping = {
    '-lrb-': '(',
    '-rrb-': ')',
    '--': '-',
    '``': '"',
    "''": '"',
}

            
def load_tagged_file(filename):
    instances = list()
    instance = list()
    with open(filename, mode="r", encoding='utf8') as f:
        for line in f:
            if line.strip():
                token = line.strip().split()[0]
                instance.append(tok_mapping.get(token, token))
            else:
                instances.append(instance)
                instance = list()
    if instance: instances.append(instance)
    return instances


def _load_spacy(pkg='en_core_web_lg'):
    try:
        nlp = spacy.load(pkg)
    except OSError as error:
        if f"Can't find model '{pkg}'" in error.args[0]:
            print(f'Downloading {pkg} from spacy servers...')
            os.system('python3 -m spacy download en_core_web_lg > /dev/null')
            print('Done. Loading parser...')
            nlp = spacy.load(pkg)
        else:
            raise error
    return nlp
            

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
    
    group = parser.add_argument_group('Arguments regarding multiprocessing')
    group.add_argument('--n_jobs', dest='n_jobs', type=int, default=-1,
                        help='number of processes to use. <0 for cpu_count()')

    args = parser.parse_args()
    
    print('Loading sentences...')
    start_load = time.time()
    if args.format == 'sent':
        with open(args.orig, mode="r", encoding='utf8') as f:
            sentences = [
                [tok_mapping.get(token, token)
                 for token in line.strip().split()]
                for line in f
            ]
    else:
        sentences = load_tagged_file(args.orig)
    time_taken = np.round(time.time() - start_load, decimals=3)
    print(f'[OK] ({len(sentences)} sentences in {time_taken} seconds)')
    
    print('Loading SpaCy parser...')
    start_load = time.time()
    nlp = _load_spacy()
    time_taken = np.round(time.time() - start_load, decimals=3)
    print(f'[OK] ({time_taken} seconds)')

    # this is the function to apply to all sentences
    def _deal_with_one_instance(sentence):
        ret = nlp.parser(Doc(nlp.vocab, sentence))
        return '\n'.join([
            f'{token.text}\t{token.dep_}\t{0 if token.dep_ == "root" else token.head.i}'
            for token in ret
        ])
    
    n_jobs = mp.cpu_count() if args.n_jobs < 0 else args.n_jobs
    print(f'Using {n_jobs} processes, starting now.')
    
    with mp.Pool(processes=n_jobs) as pool:
        processed_sentences = [item for item in tqdm.tqdm(
            pool.imap(
                _deal_with_one_instance, 
                sentences,
                chunksize=10
            ),
            desc='Parsing sentences',
            total=len(sentences)
        )]
    
    print('Serializing results, one token + tags per line',
          '+ empty line to separate sentences')
    with open(args.dest, mode='w', encoding='utf8') as f_tags:
        for sentence in processed_sentences:
            f_tags.write(sentence + '\n\n')
