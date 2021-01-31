import argparse
import json

from nltk.translate.bleu_score import corpus_bleu
from data.utils import FileIterable
from parent.parent import parent
from tqdm import tqdm


def _corpus_bleu(hypotheses, references):
    return corpus_bleu([[r for r in refs if r] for refs in zip(*references)], hypotheses)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute BLEU and PARENT metrics.')
    parser.add_argument('--tables', dest="tables")
    parser.add_argument('--references', dest="references", nargs='+')
    parser.add_argument('--hypotheses', dest="hypotheses")

    args = parser.parse_args()
    
    tables = FileIterable.from_filename(args.tables, fmt='jl').to_list('Reading Tables')
    references = [FileIterable.from_filename(filename, fmt='txt').to_list(f'Reading References {idx}')
                  for idx, filename in enumerate(args.references, 1)]
    hypotheses = FileIterable.from_filename(args.hypotheses, fmt='txt').to_list('Reading Predictions')
    
    print('Computing BLEU... ', end='')
    bleu = _corpus_bleu(hypotheses, references)
    print('OK')

    print('Computing PARENT... ', end='')
    #references = [r[0] for r in references]
    parent_p, parent_r, parent_f = parent(hypotheses, references, tables)
    print('OK')

    print(f'\n{args.hypotheses}:\nBLEU\t{bleu:.4f}\n'
          f'PARENT (precision)\t{parent_p:.4f}\n'
          f'PARENT (recall)\t{parent_r:.4f}\n'
          f'PARENT (F1)\t{parent_f:.4f}\n')
