import argparse
import functools
import os
import subprocess

import pkg_resources
import pyconll

partial_shell = functools.partial(subprocess.run, shell=True,
                                  stdout=subprocess.PIPE)


def shell(cmd):
    """Execute cmd as if from the command line"""
    completed_process = partial_shell(cmd)
    return completed_process.stdout.decode("utf8")


def read_conllu(path):
    """
    Read the conll formatted file,
    and return sentences as lists of tuples (word, deprel)
    """

    data = pyconll.load_from_file(path)
    tagged_sentences = []
    t = 0
    for sentence in data:
        tagged_sentence = []
        for token in sentence:
            if token.deprel and token.form:
                t += 1
                deprel = token.deprel.split(':')[0]
                tagged_sentence.append((token.form.lower(), token.upos, deprel, token.head))
        tagged_sentences.append(tagged_sentence)
    return tagged_sentences


def do_train(folder, gpus):
    """
    Fine-tune BERT on UniversalDependencies. If needed, download and format the training set.
    """

    # maybe download the training dataset
    if not os.path.exists(folder):
        os.mkdir(folder)

    file = dict()
    for subset in ['train', 'dev']:
        file_dl = os.path.join(folder, f'en_partut-ud-{subset}.conllu')

        if not os.path.exists(file_dl):
            print(f'No pre-existing {subset} file found, downloading.')
            shell(
                f'wget https://raw.githubusercontent.com/UniversalDependencies/UD_English-ParTUT/master/en_partut-ud-{subset}.conllu')
            shell(f'mv en_partut-ud-{subset}.conllu {folder}/')

        file[subset] = os.path.join(folder, f'{subset}.txt')
        if not os.path.exists(file[subset]):
            print(f'Formatting {subset} file, one word per line')

            sentences = read_conllu(os.path.join(folder, f'en_partut-ud-{subset}.conllu'))
            with open(file[subset], mode='w', encoding='utf8') as f:
                for sentence in sentences:
                    for token, pos, deprel, head in sentence:
                        f.write(f'{token} {pos} {deprel} {head}\n')
                    f.write('\n')

    label_file = os.path.join(folder, 'labels.txt')
    if not os.path.exists(label_file):
        print('Extracting unique labels in labels.txt')
        shell(f'cat {file["train"]} | cut -d " " -f 2 | grep -v "^$"| sort | uniq > {label_file}')

    env_variables = {
        'CUDA_VISIBLE_DEVICES': gpus
    }
    env_variables = ' '.join([f'{key}={value}' for key, value in env_variables.items()])

    print('Using the following environment variables, please edit the script if needed')
    print(env_variables)

    training_args = {
        'data_dir': f'{folder}',
        'model_type': 'bert_deprel',
        'labels': f'{folder}/labels.txt',
        'model_name_or_path': 'bert-base-uncased',
        'output_dir': f'{folder}/trained',
        'max_seq_length': '256',
        'num_train_epochs': '5',
        'per_gpu_train_batch_size': '32',
        'save_steps': '750',
        'logging_steps': '56',
        'label_size': '2'
    }

    training_args = ' '.join([f'--{key} {value}' for key, value in training_args.items()])
    print('Using the following arguments, please edit the script if needed')
    print(training_args)

    shell(f'{env_variables} python3 run_ner.py {training_args} --do_train --evaluate_during_training')


def run_script(examples, deprel_folder, wiki_folder, setname, gpus):
    # This dict is used to map the weirdly formatted tokens of wikibio
    # to tokens known to BERT
    tok_mapping = {
        '-lrb-': '(',
        '-rrb-': ')',
        '--': '-',
        '``': '"',
        "''": '"',
    }

    path = os.path.join(deprel_folder, 'test.txt')
    with open(path, mode='w', encoding='utf8') as f:
        for example in examples:
            for token in example.split():
                f.write(f'{tok_mapping.get(token, token)}\n')
            f.write('\n')

    cmd = " ".join([
        f'CUDA_VISIBLE_DEVICES={gpus}',
        'python3 run_ner.py',
        f'--data_dir {deprel_folder}/',
        '--model_type bert_deprel',
        f'--labels {os.path.join(deprel_folder, "labels.txt")}',
        '--model_name_or_path bert-base-uncased',
        f'--output_dir {os.path.join(deprel_folder, "trained")}',
        '--max_seq_length 256',
        '--do_predict',
        '--per_gpu_eval_batch_size 64',
        '--label_size 2',
        '--input_size 2'
    ])
    shell(cmd)

    dest = os.path.join(wiki_folder, f'{setname}_deprel.txt')
    if not os.path.exists(dest):
        with open(dest, mode="w", encoding='utf8') as f:
            pass

    orig = os.path.join(deprel_folder, 'trained', 'test_predictions.txt')
    with open(dest, mode="a", encoding='utf8') as destfile, \
            open(orig, mode="r", encoding='utf8') as origfile:
        for line in origfile:
            destfile.write(line.strip() + "\n")

    # removing cached data so that we can continue training on different examples
    shell(f'rm {os.path.join(deprel_folder, "cached_test_bert-base-uncased_256")}')


def do_tagging(deprel_folder, wiki_folder, setnames, gpus, split_size=int(5e4)):
    """This will format the train/dev/test sets of wikibio 
    so that we can run the deprel tagging network we have trained"""

    for setname in setnames:
        assert setname in ['train', 'valid', 'test']
        print(f'Loading examples from {setname}')

        path = os.path.join(wiki_folder, f'{setname}_output.txt')
        examples = list()
        with open(path, mode='r', encoding='utf8') as f:
            for line in f:
                if not line.strip(): continue
                examples.append(line.strip())

                if 0 < split_size <= len(examples):
                    run_script(examples, deprel_folder, wiki_folder, setname, gpus)
                    examples = list()

        if examples:
            run_script(examples, deprel_folder, wiki_folder, setname, gpus)
        print('Done.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', dest='do_train', action='store_true',
                        help='fine-tune BERT with a deprel-tagging task')
    parser.add_argument('--do_tagging', dest='do_tagging', nargs='+',
                        help='use fine-tuned BERT to tag sentences in WikiBIO.'
                             'Specify any combinason of [train, valid, test]')
    parser.add_argument('--gpus', dest='gpus', nargs="+", type=int,
                        help='list of devices to train/predict on.')
    parser.add_argument('--split_size', dest='split_size', type=int, default=5e4,
                        help='To be memory efficient, process this much line at once only.')

    args = parser.parse_args()

    deprel_folder = pkg_resources.resource_filename(__name__, 'deprel')
    wiki_folder = pkg_resources.resource_filename(__name__, 'wikibio')

    gpus = ','.join(map(str, args.gpus))
    if not gpus:
        print('Not using gpu can be significantly slower.')
        print('You can specify devices using --gpu 0 1 2 3 for example.')
    else:
        print(f'Using the following device{"s" if len(args.gpus) > 1 else ""}: [{gpus}]')

    if args.do_train:
        do_train(deprel_folder, gpus)

    if args.do_tagging:
        do_tagging(deprel_folder, wiki_folder, args.do_tagging, gpus, args.split_size)
