"""
This script is mendatory before training a PoS-tagger.
Indeed, hugginface/transformers/examples scripts have a max-length
feature, used to truncate all sentences exceeding this size. 

Therefore we remove sentences that are too long, to be sure no
sentence is truncated. Manual exploration found that for any size
above 256 tokens, no sentence from the test set will be removed.
"""


from transformers import BertTokenizer
from tqdm import tqdm

import pkg_resources
import argparse
import os


def count_lines_in_file(path):
    with open(path, mode="r", encoding="utf8") as f:
        idx = 0
        for idx, _ in enumerate(f): pass
    return idx + 1 if idx else 0


def main(args):    
    
    # This mapping ensures weird wikibio chars are read by BERT tokenizers
    tok_mapping = {
        '-lrb-': '(',
        '-rrb-': ')',
        '--': '-',
        '``': '"',
        "''": '"',
    }
    
    # choose the tokeniser that will be used in PoS tagging later
    if 'bert' in args.model.lower():
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        raise RuntimeError('Please use BERT for now')
        
        
    folder = pkg_resources.resource_filename(__name__, 'wikibio')
    
    for setname in ['train', 'valid', 'test']:
        
        print(f'Dealing with {setname}')
        
        indices_to_remove = set()
        
        # paths
        full_output_path = os.path.join(folder, 'full', f'{setname}_output.txt')
        final_output_path = os.path.join(folder, f'{setname}_output.txt')
        
        # Counting lines (iterating the file with noop) for tqdm
        nlines = count_lines_in_file(full_output_path)
        print(f'\tDealing with output ({nlines} lines)')
        
        with open(full_output_path, mode="r", encoding='utf8') as f, \
                open(final_output_path, mode="w", encoding='utf8') as g:
            
            for idx, line in tqdm(enumerate(f), total=nlines):
                tmp = ' '.join([tok_mapping.get(tok, tok)
                                for tok in line.strip().split()])
                
                if len(tokenizer.encode(tmp)) > args.max_size:
                    indices_to_remove.add(idx)
                else:
                    g.write(line.strip() + '\n')
                    
        print(f'\tFound {len(indices_to_remove)} sentences to remove')
                    
        for suffix in ['input.txt', 'tables.jl']:
            
            print(f'\tDealing with {suffix}')
            
            full_output_path = os.path.join(folder, 'full', f'{setname}_{suffix}')
            final_output_path = os.path.join(folder, f'{setname}_{suffix}')
            with open(full_output_path, mode="r", encoding='utf8') as f, \
                    open(final_output_path, mode="w", encoding='utf8') as g:

                for idx, line in tqdm(enumerate(f), total=nlines):
                    if idx in indices_to_remove:
                        continue
                    else:
                        g.write(line.strip() + '\n')
    print('Done.')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', dest='model', default='bert-base-uncased',
                        help='The hugginface/tokenizers tokeniser name')
    parser.add_argument('--max_size', dest='max_size', type=int, default=256,
                        help='The max tolerated size')
    
    args = parser.parse_args()
    
    main(args)
