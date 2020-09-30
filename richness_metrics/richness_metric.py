from collections import defaultdict
import numpy as np
import spacy
import codecs
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import argparse
import json
import os
import multiprocessing as mp
import tqdm


def calc_tfidf(nlp, threshold=0.005):
    keystr_list = codecs.open("test.box.lab", "r", encoding="utf-8").read().strip().split('\n')
    valstr_list = codecs.open("test.box.val", 'r', encoding="utf-8").read().strip().split('\n')
    wordidf_count, attr_count = defaultdict(int), defaultdict(int)
    num_tables = len(keystr_list)

    for i, (keystr, valstr) in enumerate(zip(keystr_list, valstr_list)):
        keys = keystr.strip().split(' ')
        vals = valstr.strip().split(' ')
        assert len(keys) == len(vals)
        for key in set(keys):
            attr_count[key] += 1
        lemma_set = set()
        for token in vals:
            token = lemmatizer.lemmatize(token).lower()
            if token not in stop_words and token.isalpha():
                lemma_set.add(token)
        for lemma in lemma_set:
            wordidf_count[lemma] += 1
        print(i)
    for word in wordidf_count:
        wordidf_count[word] = np.log10(num_tables / wordidf_count[word])
    for key in attr_count:
        if attr_count[key] < threshold * num_tables:
            attr_count[key] = 0
        else:
            attr_count[key] = 1.0 / attr_count[key]

    return wordidf_count, attr_count



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", default="test_summary.txt", type=str)
    # if the frequency of certain attribute is smaller than threshold, we ignore the content in that attribute
    parser.add_argument("--threshold", default=0.005, type=float)
    args = parser.parse_args()
    
    print('loading models')
    nlp = spacy.load("en_core_web_sm")
    nlp.tokenizer = nlp.tokenizer.tokens_from_list
    stop_words = codecs.open("stopwords.txt", "r", encoding="utf-8").read().strip().split('\n')
    lemmatizer = WordNetLemmatizer()

    
    
    print('Computing tfidf')
    if os.path.exists('word.json') and os.path.exists('attr.json'):
        with open('word.json', mode='r') as f:
            wordidf_count = json.load(f)
        with open('attr.json', mode='r') as f:
            attr_count = json.load(f)
    
    else:
        wordidf_count, attr_count = calc_tfidf(nlp)
        with open('word.json', mode='w') as f:
            json.dump(wordidf_count, f)
        with open('attr.json', mode='w') as f:
            json.dump(attr_count, f)
    
    
    print('Computing richness')
    
    keystr_list = codecs.open("test.box.lab", "r", encoding="utf-8").read().strip().split('\n')
    valstr_list = codecs.open("test.box.val", 'r', encoding="utf-8").read().strip().split('\n')
    predstr_list = codecs.open(args.pred_path, "r", encoding="utf-8").read().strip().split('\n')
    
    def _calc_richness(zipped_inputs):
        
        keystr, valstr, predstr = zipped_inputs
        
        wordtfidf_count = defaultdict(int)
        keys = keystr.strip().split(' ')
        vals = valstr.strip().split(' ')
        tokens = word_tokenize(predstr)

        assert len(keys) == len(vals)
        for idx, token in enumerate(vals):
            token = lemmatizer.lemmatize(token).lower()
            if token not in stop_words and token.isalpha():
                wordtfidf_count[token] += wordidf_count[token] * attr_count[keys[idx]]

        total_sum = sum(wordtfidf_count.values())
        cum_sum = 0
        for token in set(tokens):
            if token.startswith("**") and token.endswith("**"):
                token = token[2:-2]
            cum_sum += wordtfidf_count[token]
        if total_sum != 0:
            return cum_sum / total_sum, 1
        else:
            return 0, 0
        
    zipped_inputs = zip(keystr_list, valstr_list, predstr_list)
        
    total_richness = 0.
    non_empty_i = 0
    
    n_jobs = mp.cpu_count()
    chunksize = n_jobs
    print(f'Using {n_jobs} processes, starting now')
    with mp.Pool(processes=n_jobs) as pool:
        _iterable = pool.imap(
            _calc_richness, 
            zipped_inputs,
            chunksize=chunksize
        )
        
        for a, b in tqdm.tqdm(
            _iterable, total=len(keystr_list), desc='Computing richness'):
            total_richness += a
            non_empty_i += b
    
    print("Information Richness of {} is {}".format(args.pred_path, total_richness / non_empty_i))
                                                    #calc_richness(args.pred_path, nlp, wordidf_count, attr_count)))
