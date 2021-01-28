from utils import FileIterable

import argparse
import tqdm
import json
import os


DELIM = u"￨"  # delim used by onmt


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    group = parser.add_argument_group('File system')
    group.add_argument('--orig', '-o', dest='orig',
                        help='Name of the tables file')
    group.add_argument('--dest', '-d', dest='dest',
                        help='Name of the inputs file.')
    
    group = parser.add_argument_group('Other')
    group.add_argument('--halwo', dest='halwo', default=None,
                       help='Add an hallucination word overlap token to each table.')
    
    args = parser.parse_args()
    

    tables = FileIterable.from_filename(args.orig, fmt='jl')
    
    with open(args.dest, mode="w", encoding='utf8') as f:
        for table in tqdm.tqdm(tables, desc=f'Creating f{args.dest}'):
            if args.halwo is not None:
                table.append(['HALWO', [f'[HALWO={args.halwo}]']])
            str_table = ' '.join([
                f'{value}{DELIM}{key}{DELIM}{idx}{DELIM}{len(values)-idx+1}'
                for key, values in table 
                for idx, value in enumerate(values, 1)
            ])
            f.write(str_table + '\n')