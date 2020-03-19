from transformers import BertTokenizer

import pkg_resources
import argparse


def main(args):
    if 'bert' in args.model.lower():
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        raise RuntimeError('Please use BERT for now')
        
        
    folder = pkg_resources.resource_filename(__name__, 'wikbio')
    
    for setname in ['train', 'valid', 'test']:
        
        indices_to_removes = set()
        
        full_output_path = os.path.join(folder, 'full', f'{setname}_output.txt')
        final_output_path = os.path.join(folder, f'{setname}_output.txt')
        with open(full_output_path, mode="r", encoding='utf8') as f, \
                open(final_output_path, mode="w", encoding='utf8') as f:
            
            for idx, line in enumerate(f):
                tmp = ' '.join([tok_mapping.get(tok, tok)
                                for tok in line.strip().split()])
                
                if len(tokenizer.tokenize(tmp)) > args.max_size:
                    indices_to_removes.add(idx)
                else:
                    g.write(line + '\n')
                    
        
        full_output_path = os.path.join(folder, 'full', f'{setname}_input.txt')
        final_output_path = os.path.join(folder, f'{setname}_input.txt')
        with open(full_output_path, mode="r", encoding='utf8') as f, \
                open(final_output_path, mode="w", encoding='utf8') as f:
            
            for idx, line in enumerate(f):
                if idx in indices_to_removes:
                    continue
                else:
                    g.write(line.strip() + '\n')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', dest='model', default='bert-base-uncased',
                        help='The hugginface/tokenizers tokeniser name')
    parser.add_argument('--max_size', dest='max_size', type=int, default=256,
                        help='The max tolerated size')
    
    args = parser.parse_args()
    
    main(args)