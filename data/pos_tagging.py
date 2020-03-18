import subprocess
import functools
import argparse
import pyconll
import json
import os
import re


partial_shell = functools.partial(subprocess.run, shell=True,
                                  stdout=subprocess.PIPE)
def shell(cmd):
    """Execute cmd as if from the command line"""
    completed_process = partial_shell(cmd)
    return completed_process.stdout.decode("utf8")


def read_conllu(path):
    """
    Read the conll formated file,
    and return sentences as lists of tuples (word, pos)
    """
    
    data = pyconll.load_from_file(path)
    tagged_sentences=[]
    t=0
    for sentence in data:
        tagged_sentence=[]
        for token in sentence:
            if token.upos and token.form:
                t+=1
                tagged_sentence.append((token.form.lower(), token.upos))
        tagged_sentences.append(tagged_sentence)
    return tagged_sentences



def do_train(folder):
    """
    Fine-tune BERT on UniversalDependencies. If needed, download and format the training set.
    """
    
    
    # maybe download the training dataset
    if not os.path.exists(folder):
        os.mkdir(folder)
        
    train_file_dl = os.path.join(folder, 'en_partut-ud-train.conllu')
    
    if not os.path.exists(train_file_dl):
        print('No pre-existing training file found, downloading.')
        shell('wget https://raw.githubusercontent.com/UniversalDependencies/UD_English-ParTUT/master/en_partut-ud-train.conllu')
        
    train_file = os.path.join(folder, 'train.txt')
    if not os.path.exists(train_file):
        print('Formatting train file, one word per line')
        
        sentences = read_conllu('data/en_partut-ud-train.conllu')
        with open('data/pos/train.txt', mode='w', encoding='utf8') as f:
            for sentence in sentences:
                for token, pos in sentence:
                    f.write(f'{token} {pos}\n')
                f.write('\n')

    label_file = os.path.join(folder, 'labels.txt')
    if not os.path.exists(label_file):
        print('Extracting unique labels in labels.txt')
        shell(f'cat train.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > {label_file}')
        
        
    env_variables = {
        'CUDA_VISIBLE_DEVICES': '0,1'
    }
    env_variables = ' '.join([f'{key}={value}' for key, value in env_variables.items()])
    
    print('Using the following environment variables, please edit the script if needed')
    print(env_variables)

    training_args = {
        'data_dir': f'{folder}',
        'model_type': 'bert',
        'labels': f'{folder}/labels.txt',
        'model_name_or_path': 'bert-base-uncased',
        'output_dir': f'{folder}/trained',
        'max_seq_length': '132',
        'num_train_epochs': '3',
        'per_gpu_train_batch_size': '32',
        'save_steps': '750',
    }
    
    training_args = ' '.join(['--{key} {value}' for key, value in training_args.items()])
    print('Using the following arguments, please edit the script if needed')
    print(training_args)

    shell(f'{env_variables} python run_ner.py {training_args} --do_train')

def do_tagging(pos_folder, wiki_folder, setnames):
    """This will format the train/dev/test sets of wikibio 
    so that we can run the PoS tagging network we have trained"""
    
    # This dict is used to map the weirdly formatted tokens of wikibio 
    # to tokens known to BERT
    tok_mapping = {
        '-lrb-': '(',
        '-rrb-': ')',
        '--': '-',
        '``': '"',
        "''": '"',
    }
    
    for setname in setnames:
        assert setname in ['train', 'valid', 'test']
        print(f'Loading examples from {setname}')
        with open(f'{wiki_folder}/{setname}_output.txt', mode='r', encoding='utf8') as f:
            examples = [line.strip() for line in f if line.strip()]
            
        print('Formating examples (one token per line)')
        with open(f'{folder}/test.txt', mode='w', encoding='utf8') as f:
            for example in examples:
                for token in example.split():
                    f.write(f'{tok_mapping.get(token, token)}\n')
                f.write('\n')
    
        print('Starting prediction of Part of Speech')
        cmd = " ".join([
            'CUDA_VISIBLE_DEVICES=0,1',
            'python run_ner.py',
            f'--data_dir {pos_folder}/',
            '--model_type bert',
            f'--labels {pos_folder}/labels.txt',
            '--model_name_or_path bert-base-uncased',
            f'--output_dir {pos_folder}/trained',
            '--max_seq_length 132',
            '--do_predict',
            '--per_gpu_eval_batch_size 64'
        ])
        shell(cmd)
        
        print('Moving prediction file to data/wikibio')
        shell(f'cp {pos_folder}/trained/test_predictions.txt {wiki_folder}/{setname}_pos.txt')
        shell(f'rm {pos_folder}/cached_test_bert-base-uncased_132')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', dest='do_train', action='store_true',
                        help='fine-tune BERT with a PoS-tagging task')
    parser.add_argument('--do_tagging', dest='do_tagging', nargs='+',
                        help='use fine-tuned BERT to tag sentences in WikiBIO.'
                             'Specify any combinason of [train, valid, test]')
    
    
    pos_folder = pkg_resources.resource_filename(__name__, 'pos')
    wiki_folder = pkg_resources.resource_filename(__name__, 'wikbio')
    
    if args.do_train:
        do_train(pos_folder)
        
    if args.do_tagging:
        do_taggin(pos_folder, wiki_folder, args.do_tagging)