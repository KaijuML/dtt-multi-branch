"""
Plots an histogram of the tokens labelled as ['NOUN', 'ADJ', 'NUM', 'PROPN'], using Pyplot.
Very memory consuming!
"""

import pickle
from collections import Counter
from os import path

import matplotlib.pyplot as plt
from tqdm import tqdm

from data.co_occurrence import pos_filename, data_folder, interesting_tags, num_words


def main():
    counter_file = path.join(data_folder, 'train_word_count.pickle')
    if path.isfile(counter_file):
        with open(counter_file, 'rb') as f:
            return pickle.load(f)

    cnt = Counter()
    with open(pos_filename) as f_refs:
        for tagged_word in tqdm(f_refs, desc='Counting word frequencies in references', total=num_words):
            if tagged_word != '\n':
                word, tag = tagged_word.split()
                if tag in interesting_tags:
                    cnt.update([word])

    with open(counter_file, 'wb') as f:
        pickle.dump(cnt, f)

    # Plot common words histogram
    cnt = dict(cnt.most_common(100))
    plt.bar(cnt.keys(), cnt.values())
    plt.xticks(rotation='vertical')
    plt.show()


if __name__ == '__main__':
    main()
