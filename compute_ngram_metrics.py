import argparse
import json

from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

from parent import parent


def read_files(_args):
    _tables, _references, _hypotheses = [], [], []
    with open(_args.tables) as tables_file, open(_args.references) as references_file, \
            open(args.hypotheses) as hypotheses_file:
        for table, reference, hypothesis in tqdm(
                zip(tables_file, references_file, hypotheses_file), desc='Reading files', unit='lines'):
            _tables.append(json.loads(table))
            _references.append([reference.split()])
            _hypotheses.append(hypothesis.split())
    _tables = list(map(lambda tab: [([row[0]], row[1]) for row in tab], _tables))
    return _tables, _references, _hypotheses


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute BLEU and PARENT metrics.')
    parser.add_argument('tables')
    parser.add_argument('references')
    parser.add_argument('hypotheses')

    args = parser.parse_args()

    tables, references, hypotheses = read_files(args)

    print('Computing BLEU... ', end='')
    bleu = corpus_bleu(references, hypotheses)
    print('OK')

    print('Computing PARENT... ', end='')
    references = [r[0] for r in references]
    parent_p, parent_r, parent_f = parent(hypotheses, references, tables)
    print('OK')

    print(f'\n{args.hypotheses}:\nBLEU\t{bleu:.4f}\n'
          f'PARENT (precision)\t{parent_p:.4f}\n'
          f'PARENT (recall)\t{parent_r:.4f}\n'
          f'PARENT (F1)\t{parent_f:.4f}\n')
