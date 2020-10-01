import pkg_resources
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


def do_train(folder, gpus):
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
        shell(f'mv en_partut-ud-train.conllu {folder}/')
        
    train_file = os.path.join(folder, 'train.txt')
    if not os.path.exists(train_file):
        print('Formatting train file, one word per line')
        
        sentences = read_conllu(os.path.join(folder, 'en_partut-ud-train.conllu'))
        with open(train_file, mode='w', encoding='utf8') as f:
            for sentence in sentences:
                for token, pos in sentence:
                    f.write(f'{token} {pos}\n')
                f.write('\n')

    label_file = os.path.join(folder, 'labels.txt')
    if not os.path.exists(label_file):
        print('Extracting unique labels in labels.txt')
        shell(f'cat {train_file} | cut -d " " -f 2 | grep -v "^$"| sort | uniq > {label_file}')
        
        
    env_variables = {
        'CUDA_VISIBLE_DEVICES': gpus
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
        'max_seq_length': '256',
        'num_train_epochs': '3',
        'per_gpu_train_batch_size': '32',
        'save_steps': '750',
    }
    
    training_args = ' '.join([f'--{key} {value}' for key, value in training_args.items()])
    print('Using the following arguments, please edit the script if needed')
    print(training_args)

    shell(f'{env_variables} python run_ner.py {training_args} --do_train')
    
    
def run_script(examples, pos_folder, dest, gpus):
    
    # This dict is used to map the weirdly formatted tokens of wikibio 
    # to tokens known to BERT
    tok_mapping = {
        '-lrb-': '(',
        '-rrb-': ')',
        '--': '-',
        '``': '"',
        "''": '"',
    }
    
    # 
    path = os.path.join(pos_folder, 'test.txt')
    with open(path, mode='w', encoding='utf8') as f:
        for example in examples:
            for token in example.split():
                f.write(f'{tok_mapping.get(token, token)}\n')
            f.write('\n')
            
    cmd = " ".join([
        f'CUDA_VISIBLE_DEVICES={gpus}',
        'python run_ner.py',
        f'--data_dir {pos_folder}/',
        '--model_type bert',
        f'--labels {os.path.join(pos_folder, "labels.txt")}',
        '--model_name_or_path bert-base-uncased',
        f'--output_dir {os.path.join(pos_folder, "trained")}',
        '--max_seq_length 256',
        '--do_predict',
        '--per_gpu_eval_batch_size 64'
    ])
    shell(cmd)

    if not os.path.exists(dest):
        with open(dest, mode="w", encoding='utf8') as f:
            pass
    
    orig = os.path.join(pos_folder, 'trained', 'test_predictions.txt')
    with open(dest, mode="a", encoding='utf8') as destfile, \
            open(orig, mode="r", encoding='utf8') as origfile:
        for line in origfile:
            destfile.write(line.strip() + "\n")
            
    # removing cached data so that we can continue training on different examples
    shell(f'rm {os.path.join(pos_folder, "cached_test_bert-base-uncased_256")}')

def do_tagging(pos_folder, wiki_folder, setnames, gpus, split_size=int(5e4)):
    """This will format the train/dev/test sets of wikibio 
    so that we can run the PoS tagging network we have trained"""
    
    for setname in setnames:
        assert setname in ['train', 'valid', 'test']
        print(f'Loading examples from {setname}')
        
        path = os.path.join(wiki_folder, f'{setname}_output.txt')
        dest = os.path.join(wiki_folder, f'{setname}_pos.txt')
        examples = list()
        with open(path, mode='r', encoding='utf8') as f:
            for line in f:
                if not line.strip(): continue
                examples.append(line.strip())
                
                if split_size>0 and len(examples) >= split_size:
                    run_script(examples, pos_folder, dest, gpus)
                    examples = list()
                    
        if examples:
            run_script(examples, pos_folder, dest, gpus)
        print('Done.')
            
def do_file(pos_folder, orig, dest, gpus, split_size=int(5e4)):
    """This will tag an individual file 'orig' into 'dest'"""
    examples = list()
    with open(orig, mode='r', encoding='utf8') as f:
        for line in f:
            if not line.strip(): continue
            examples.append(line.strip())

            if split_size>0 and len(examples) >= split_size:
                run_script(examples, pos_folder, dest, gpus)
                examples = list()

    if examples:
        run_script(examples, pos_folder, dest, gpus)
    print('Done.')

if __name__ == '__main__':
    pos_folder = pkg_resources.resource_filename(__name__, 'pos')
    wiki_folder = pkg_resources.resource_filename(__name__, 'wikibio')

    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', dest='do_train', action='store_true',
                        help='fine-tune BERT with a PoS-tagging task')
    parser.add_argument('--do_tagging', dest='do_tagging', nargs='+',
                        help='use fine-tuned BERT to tag sentences in WikiBIO.'
                             'Specify any combinason of [train, valid, test]')
    parser.add_argument('--gpus', dest='gpus', nargs="+", type=int, 
                        help='list of devices to train/predict on.')
    parser.add_argument('--split_size', dest='split_size', type=int, default=5e4,
                        help='To be memory efficient, process this much line at once only.')
    parser.add_argument('--pos_folder', dest='pos_folder', default=pos_folder)
    parser.add_argument('--wiki_folder', dest='wiki_folder', default=wiki_folder)

    # These arguments are for stand-alone file
    parser.add_argument('--orig', '-o', dest='orig',
                        help='Name of the stand alone file')
    parser.add_argument('--dest', '-d', dest='dest',
                        help='Name of the resulting file')
    
    args = parser.parse_args()

    gpus = ','.join(map(str, args.gpus))
    if not gpus:
        print('Not using gpu can be significantly slower.')
        print('You can specify devices using --gpu 0 1 2 3 for example.')
    else:
        print(f'Using the following device{"s" if len(args.gpus)>1 else ""}: [{gpus}]' )
    
    if args.do_train:
        do_train(args.pos_folder, gpus)
        
    if args.do_tagging:
        do_tagging(args.pos_folder, args.wiki_folder, args.do_tagging, gpus, args.split_size)
        
    if args.orig:
        assert os.path.exists(args.orig)
        do_file(args.pos_folder, args.orig, args.dest, gpus, args.split_size)