from utils import nwise

import pkg_resources
import re, time, os
import itertools
import argparse
import numpy
import json

DELIM = u"￨"  # delim used by onmt


def split_infobox(dataset_folder, destination_folder):
    """
    extract box content, field type and position information from infoboxes from original_data
    *.box.val is the box content (token)
    *.box.lab is the field type for each token
    *.box.pos is the position counted from the begining of a field
    """
    bwfile = [os.path.join(destination_folder, 'processed_data', setname, f"{setname}.box.val")
              for setname in ['train', 'valid', 'test']]
    bffile = [os.path.join(destination_folder, 'processed_data', setname, f"{setname}.box.lab")
              for setname in ['train', 'valid', 'test']]
    bpfile = [os.path.join(destination_folder, 'processed_data', setname, f"{setname}.box.pos")
              for setname in ['train', 'valid', 'test']]

    mixb_word, mixb_label, mixb_pos = [], [], []
    for setname in ['train', 'valid', 'test']:
        fboxes = os.path.join(dataset_folder, 'raw', setname, f"{setname}.box")
        with open(fboxes, mode="r", encoding="utf8") as f:
            box = [line.strip() for line in f if line.strip()]
        box_word, box_label, box_pos = [], [], []
        for ib in box:
            item = ib.split('\t')
            box_single_word, box_single_label, box_single_pos = [], [], []
            for it in item:
                if len(it.split(':')) > 2:
                    continue
                # print it
                prefix, word = it.split(':')
                if '<none>' in word or word.strip() == '' or prefix.strip() == '':
                    continue
                new_label = re.sub("_[1-9]\d*$", "", prefix)
                if new_label.strip() == "":
                    continue
                box_single_word.append(word)
                box_single_label.append(new_label)
                if re.search("_[1-9]\d*$", prefix):
                    field_id = int(prefix.split('_')[-1])
                    box_single_pos.append(field_id if field_id <= 30 else 30)
                else:
                    box_single_pos.append(1)
            box_word.append(box_single_word)
            box_label.append(box_single_label)
            box_pos.append(box_single_pos)
        mixb_word.append(box_word)
        mixb_label.append(box_label)
        mixb_pos.append(box_pos)

        print(f'{setname} done')

    for k, m in enumerate(mixb_word):
        with open(bwfile[k], "w+") as h:
            for items in m:
                for sens in items:
                    h.write(str(sens) + " ")
                h.write('\n')
    for k, m in enumerate(mixb_label):
        with open(bffile[k], "w+") as h:
            for items in m:
                for sens in items:
                    h.write(str(sens) + " ")
                h.write('\n')
    for k, m in enumerate(mixb_pos):
        with open(bpfile[k], "w+") as h:
            for items in m:
                for sens in items:
                    h.write(str(sens) + " ")
                h.write('\n')


def reverse_pos(folder):
    # get the position counted from the end of a field
    bpfile = [os.path.join(folder, 'processed_data', setname, f"{setname}.box.pos")
              for setname in ['train', 'valid', 'test']]
    bwfile = [os.path.join(folder, 'processed_data', setname, f"{setname}.box.rpos")
              for setname in ['train', 'valid', 'test']]

    for k, pos in enumerate(bpfile):
        box = open(pos, "r").read().strip().split('\n')
        reverse_pos = []
        for bb in box:
            pos = bb.split()
            tmp_pos = []
            single_pos = []
            for p in pos:
                if int(p) == 1 and len(tmp_pos) != 0:
                    single_pos.extend(tmp_pos[::-1])
                    tmp_pos = []
                tmp_pos.append(p)
            single_pos.extend(tmp_pos[::-1])
            reverse_pos.append(single_pos)
        with open(bwfile[k], 'w+') as bw:
            for item in reverse_pos:
                bw.write(" ".join(item) + '\n')


def create_input(folder):
    for setname in ["train", "valid", "test"]:

        valfilename = os.path.join(folder, 'processed_data', setname, f"{setname}.box.val")
        labfilename = os.path.join(folder, 'processed_data', setname, f"{setname}.box.lab")
        posfilename = os.path.join(folder, 'processed_data', setname, f"{setname}.box.pos")
        rposfilename = os.path.join(folder, 'processed_data', setname, f"{setname}.box.rpos")

        with open(valfilename, mode='r', encoding='utf8') as valfile:
            vals = [line.strip() for line in valfile if line.strip()]
        with open(labfilename, mode='r', encoding='utf8') as labfile:
            labs = [line.strip() for line in labfile if line.strip()]
        with open(posfilename, mode='r', encoding='utf8') as posfile:
            poss = [line.strip() for line in posfile if line.strip()]
        with open(rposfilename, mode='r', encoding='utf8') as rposfile:
            rposs = [line.strip() for line in rposfile if line.strip()]

        assert len(vals) == len(labs) == len(poss) == len(rposs)

        input = list()
        for idx, (val, lab, pos, rpos) in enumerate(zip(vals, labs, poss, rposs)):
            vval = val.strip().split(' ')
            llab = lab.strip().split(' ')
            ppos = pos.strip().split(' ')
            rrpos = rpos.strip().split(' ')

            if not len(vval) == len(llab) == len(ppos) == len(rrpos):
                print(f"error at step {idx}:", len(vval), len(llab), len(ppos), len(rrpos))
                raise RuntimeError

            input.append(
                ' '.join([re.sub('\s', '~', DELIM.join(tup))
                          for tup in zip(vval, llab, ppos, rrpos)])
            )

        input_filename = os.path.join(folder, 'full', f"{setname}_input.txt")
        with open(input_filename, mode="w", encoding="utf8") as f:
            for i in input:
                f.write(i + "\n")

        print(f'{setname} done.')


def extract_sentences(dataset_folder, destination_folder, only_first=True):
    for setname in ['train', 'valid', 'test']:
        inputnb_filename = os.path.join(dataset_folder, 'raw', setname, f"{setname}.nb")
        inputsent_filename = os.path.join(dataset_folder, 'raw', setname, f"{setname}.sent")
        output_filename = os.path.join(destination_folder, 'full', f"{setname}_output.txt")

        nb = [0]
        with open(inputnb_filename, encoding='utf8', mode='r') as f:
            # Here we get the indices of the first sentence for each instance
            # The file .sent contains one sentence per line but instances have
            # more than one sentence each. (number is variable)
            for idx, line in enumerate(f):
                nb += [int(line.strip())]
        indices = numpy.cumsum(nb[:-1])

        sentences = list()
        with open(inputsent_filename, encoding='utf8', mode='r') as f:
            for idx, line in enumerate(f):
                sentences += [line.strip()]

        if only_first:
            with open(output_filename, mode='w', encoding='utf8') as f:
                for idx in indices:
                    f.write(sentences[idx] + '\n')
        else:
            with open(output_filename, mode='w', encoding='utf8') as f:
                for start, end in nwise(indices, n=2):
                    f.write(' '.join(sentences[start:end]) + '\n')


def create_tables(folder):
    """Here we create the tables.jl files used in PARENT metric
    We could optimize the code so that step is done in create_input
    but it's easier and more convienient to just add it there.
    """
    for setname in ['train', 'valid', 'test']:
        input_filename = os.path.join(folder, 'full', f"{setname}_input.txt")
        with open(input_filename, mode="r", encoding="utf8") as f:
            # each line is a table. Each token is a value in the table.
            # We take the value/label of the token and discard the pos
            # given that they are written in the right order

            allvals = list()
            alllabs = list()

            for line in f:
                vals = list()
                labs = list()
                for token in line.strip().split():
                    val, lab, _, __ = token.split(DELIM)
                    vals.append(val)
                    labs.append(lab)
                allvals.append(vals)
                alllabs.append(labs)

            tables = list()
            for idx, (vals, labs) in enumerate(zip(allvals, alllabs)):
                table = list()
                for key, group in itertools.groupby(labs):
                    size = len([_ for _ in group])
                    vvals, vals = vals[:size], vals[size:]
                    table.append((key, vvals))

                assert len(vals) == 0  # we exhausted all tokens
                tables.append(table)

        output_filename = os.path.join(folder, 'full', f"{setname}_tables.jl")
        with open(output_filename, mode="w", encoding="utf8") as f:
            for table in tables: f.write(json.dumps(table) + '\n')


def preprocess(dataset_folder, destination_folder, args):
    """
    We use a triple <f, p+, p-> to represent the field information of a token in the specific field. 
    p+&p- are the position of the token in that field counted from the begining and the end of the field.
    For example, for a field (birthname, Jurgis Mikelatitis) in an infobox, we represent the field as
    (Jurgis, <birthname, 1, 2>) & (Mikelatitis, <birthname, 2, 1>)
    """
    print("extracting token, field type and position info from original data ...")
    time_start = time.time()
    split_infobox(dataset_folder, destination_folder)
    reverse_pos(destination_folder)
    duration = time.time() - time_start
    print(f"extract finished in {duration:.3f} seconds")

    print("merging everything into single input file ...")
    time_start = time.time()
    create_input(destination_folder)
    duration = time.time() - time_start
    print(f"merge finished in {duration:.3f} seconds")

    print("extracting first sentences from original data ...")
    time_start = time.time()
    extract_sentences(dataset_folder, destination_folder, args.first_sentence)
    duration = time.time() - time_start
    print(f"extract finished in {duration:.3f} seconds")

    print("formatting input in human readable format ...")
    time_start = time.time()
    create_tables(destination_folder)
    duration = time.time() - time_start
    print(f"formatting finished in {duration:.3f} seconds")


def make_dirs(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)

    os.mkdir(os.path.join(folder, 'full'))
    os.mkdir(os.path.join(folder, "processed_data/"))
    os.mkdir(os.path.join(folder, "processed_data/train/"))
    os.mkdir(os.path.join(folder, "processed_data/test/"))
    os.mkdir(os.path.join(folder, "processed_data/valid/"))


def main(args):
    make_dirs(args.dest)
    preprocess(dataset_folder, args.dest, args)


if __name__ == '__main__':
    dataset_folder = pkg_resources.resource_filename(__name__, 'wikibio')

    parser = argparse.ArgumentParser()
    group = parser.add_argument_group('Destination path')
    group.add_argument('--dest', '-d', dest='dest',
                       default=dataset_folder,
                       help='Folder where to store the resulting files')
    parser.add_argument('--first_sentence', action='store_true',
                        help="Activate to keep only the first sentence")

    main(parser.parse_args())
