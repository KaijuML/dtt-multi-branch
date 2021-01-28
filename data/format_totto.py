"""the method `create_table` is adapted from 
https://github.com/google-research/language/tree/master/language/totto"""

from nltk.tokenize import word_tokenize
from utils import FileIterable

import argparse
import tqdm
import json
import os


DELIM = u"￨"  # delim used by onmt


def create_table(json_example, train=True):
    
    table_json = json_example["table"]
    cell_indices = json_example["highlighted_cells"]
    table_page_title = json_example["table_page_title"]
    table_section_title = json_example["table_section_title"]
    table_section_text = json_example["table_section_text"]

    table = list()
    
    # Table values.
    for (row_index, col_index) in cell_indices:
        cell = table_json[row_index][col_index]
        value = cell['value'].strip()
        if value:
            if cell['is_header']:
                attribute = 'header'
            else:
                if len(table_json[0]) > col_index:
                    attribute=table_json[0][col_index]['value']
                else:
                    continue
            value = value.replace("|", "-")
            entry = ['_'.join(attribute.split()), word_tokenize(value)]
            table.append(entry)

    # Page title.
    if table_page_title:
        table_page_title = table_page_title.replace("|", "-")
        entry = ["page_title", word_tokenize(table_page_title)]
        table.append(entry)

    # Section title.
    if table_section_title:
        table_section_title = table_section_title.replace("|", "-")
        entry = ["section_title", word_tokenize(table_section_title)]
        table.append(entry)
    
    # Include Section text for training
    if table_section_text and train:
        table_section_text = table_section_text.replace("|", "-")
        entry = ["section_text", word_tokenize(table_section_text)]
        table.append(entry)

    return table


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    group = parser.add_argument_group('File system')
    group.add_argument('--orig', '-o', dest='orig',
                        help='Name of the folder where ToTTo train/dev are.')
    group.add_argument('--dest', '-d', dest='dest',
                        help='Name of the folder to store preocessed data.')
    
    args = parser.parse_args()
    
    for subset in ['dev', 'train']:
        
        src_filename = os.path.join(args.orig, 
                                    f'totto_{subset}_data.jsonl')
        
        no_overlap_filename = os.path.join(args.dest, f'{subset}_tables.jl')
        overlap_filename = os.path.join(args.dest, f'{subset}_overlap_tables.jl')
        raw_data = list()
        with open(overlap_filename, mode="w", encoding='utf8') as overlap_f, open(no_overlap_filename, mode="w", encoding='utf8') as no_overlap_f:

            desc = f'Formating tables of ToTTo/{subset}'
            overlap_tables, no_overlap_tables = list(), list()
            for json_example in tqdm.tqdm(FileIterable.from_filename(src_filename, fmt='jl'), desc=desc):
                raw_data.append(json_example)
                
                table = create_table(json_example, subset=='train')
                
                if json_example.get('overlap_subset', False):
                    overlap_tables.append(table)
                    overlap_f.write(json.dumps(table)+'\n')
                else:
                    no_overlap_tables.append(table)
                    no_overlap_f.write(json.dumps(table)+'\n')
                
                
            
        for suffix, tables in zip(['', '_overlap'], [no_overlap_tables, overlap_tables]):
            filename = os.path.join(args.dest, f'{subset}{suffix}_input.txt')
            with open(filename, mode="w", encoding='utf8') as f:
                for table in tqdm.tqdm(tables, desc=f'Creating {subset}{suffix}_input.txt'):
                    str_table = ' '.join([
                        f'{value}{DELIM}{key}{DELIM}{idx}{DELIM}{len(values)-idx+1}'
                        for key, values in table 
                        for idx, value in enumerate(values, 1)
                    ])
                    f.write(str_table + '\n')
                
        
        max_n_references = 0
        
        for clean, ref_key in zip(['', '_clean'], ['original_sentence', 'final_sentence']):
        
            overlap_refs, no_overlap_refs = list(), list()

            for json_example in tqdm.tqdm(raw_data, desc='Reading all references'):
                refs = [' '.join(word_tokenize(ref[ref_key]))
                        for ref in json_example['sentence_annotations']]
                if json_example.get('overlap_subset', False):
                    overlap_refs.append(refs)
                else:
                    no_overlap_refs.append(refs)
                max_n_references = max(max_n_references, len(refs))

            print(f'Found at most {max_n_references} references for a single input.')

            for suffix, references in zip(['', '_overlap'], [no_overlap_refs, overlap_refs]):
                if not len(references): continue
                for k in range(1, max_n_references+1):
                    output_filename = os.path.join(args.dest, f'{subset}{suffix}{clean}_output_{k}.txt')
                    with open(output_filename, mode="w", encoding='utf8') as f:
                        for refs in tqdm.tqdm(references, desc=f'Writting {k}th references'):
                            ref = refs[k-1] if k <= len(refs) else ''
                            f.write(ref + '\n')
                
        