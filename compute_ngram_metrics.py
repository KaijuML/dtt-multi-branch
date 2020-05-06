import argparse
import json

from nltk.translate.bleu_score import corpus_bleu
from data.utils import FileIterable
from parent import parent
from tqdm import tqdm


def _corpus_bleu(hypotheses, references, _):
    return corpus_bleu([[ref] for ref in references], hypotheses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute BLEU and PARENT metrics.')
    parser.add_argument('tables')
    parser.add_argument('references')
    parser.add_argument('hypotheses')

    args = parser.parse_args()
    
    tables = FileIterable.from_filename(args.tables, fmt='jl')
    references = FileIterable.from_filename(args.references, fmt='txt')
    hypotheses = FileIterable.from_filename(args.hypotheses, fmt='txt')
    
    zipped_inputs = [
        item for item in tqdm(
            zip(hypotheses, references, tables),
            desc='Reading files',
            total=len(tables)
        )
    ]
    

    print('Computing BLEU... ', end='')
    bleu = _corpus_bleu(*zip(*zipped_inputs))
    print('OK')

    print('Computing PARENT... ', end='')
    references = [r[0] for r in references]
    parent_p, parent_r, parent_f = parent(*zip(*zipped_inputs))
    print('OK')

    print(f'\n{args.hypotheses}:\nBLEU\t{bleu:.4f}\n'
          f'PARENT (precision)\t{parent_p:.4f}\n'
          f'PARENT (recall)\t{parent_r:.4f}\n'
          f'PARENT (F1)\t{parent_f:.4f}\n')
