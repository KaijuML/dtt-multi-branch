"""
Parses a dataset's *_output.txt file, using various NLP libraries.
"""
import argparse
import time
from math import ceil

import numpy as np
import torch
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _load_allennlp(_args):
    """
    An implementation of a neural model for dependency parsing using biaffine classifiers on top of a bidirectional LSTM
    based on 'Deep Biaffine Attention for Neural Dependency Parsing' (Dozat, 2017).
    PTB
    UAS 94.81%
    LAS 92.86%
    """
    from allennlp.predictors.predictor import Predictor
    from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
    predictor = Predictor.from_path(
        "https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz",
        cuda_device=device.index if device.type == 'cuda' else -1
    )
    predictor._tokenizer = SpacyWordSplitter(pos_tags=True, split_on_spaces=True)
    return predictor


def _load_stanza(_args):
    """
    Bi-LSTM-based deep biaffine neural dependency parser (Dozat and Manning, 2017), augmented with two linguistically
    motivated features: one that predicts the linearization order of two words in a given language, and the other that
    predicts the typical distance in linear order between them.
    UD English EWT
    UAS 86.22%
    LAS 83.59%
    """
    import stanza
    try:
        model = stanza.Pipeline(
            lang="en", processors='tokenize,pos,lemma,depparse', depparse_batch_size=_args.batch_size)
    except Exception:
        stanza.download('en')
        model = stanza.Pipeline(
            lang="en", processors='tokenize,pos,lemma,depparse', depparse_batch_size=_args.batch_size)
    model.processors['tokenize'].config['pretokenized'] = True
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # These arguments are for stand-alone file
    parser.add_argument('--orig', '-o', dest='orig', required=True,
                        help='Name of the stand alone file')
    parser.add_argument('--dest', '-d', dest='dest', required=True,
                        help='Name of the resulting file')
    parser.add_argument('--parser', '-p', choices=['allennlp', 'stanza'], required=True,
                        help='The parser to be used')
    parser.add_argument('--bsize', '-b', dest='batch_size', default=1, type=int,
                        help='Batch size')
    parser.add_argument('--format', '-f', choices=['word', 'sent'], default='sent',
                        help='Format of the orig file:\n\t'
                             '"word" means one word per line (only the first one is considered)\n\t'
                             '"sent" means one sentence per line, and words will be identified using .split()')
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
    if args.parser == 'allennlp':
        sentences = [{'sentence': ' '.join(sentence)} for sentence in sentences]
    time_taken = np.round(time.time() - start_load, decimals=3)
    print(f'[OK] ({len(sentences)} sentences in {time_taken} seconds)')

    print(f'Loading parser {args.parser}...')
    start_load = time.time()
    nlp = {
        'allennlp': _load_allennlp,
        'stanza': _load_stanza
    }[args.parser](args)
    time_taken = np.round(time.time() - start_load, decimals=3)
    print(f'[OK] ({time_taken} seconds)')

    if args.parser == 'allennlp':
        processed_sentences = []
        for batch in tqdm(chunks(sentences, args.batch_size),
                          desc='Parsing sentences',
                          total=ceil(len(sentences) / args.batch_size)):
            processed_sentences += [
                '\n'.join([
                    f'{text}\t{dep}\t{head - 1}'
                    for text, dep, head in zip(ret['words'], ret['predicted_dependencies'], ret['predicted_heads'])
                ])
                for ret in nlp.predict_batch_json(batch)]
    elif args.parser == 'stanza':
        processed_sentences = [
            '\n'.join([
                f'{token.text}\t{token.deprel}\t{token.head - 1}'
                for token in ret.words])
            for ret in nlp(sentences).sentences]
    else:
        raise NotImplementedError

    print('Serializing results, one token + tags per line',
          '+ empty line to separate sentences')
    with open(args.dest, mode='w', encoding='utf8') as f_tags:
        for sentence in processed_sentences:
            f_tags.write(sentence + '\n\n')
