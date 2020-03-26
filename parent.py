# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
# I have just rewritten stuff so that it works without tensorflow
# I also added multi-precessing to greatly speedup the computations

"""Script to compute PARENT metric."""
from functools import partial

import collections, itertools, math
import multiprocessing as mp


def overlap_probability(ngram, table, smoothing=0.0, stopwords=None):
    """Returns the probability that the given n-gram overlaps with the table.

    A simple implementation which checks how many tokens in the n-gram are also
    among the values in the table. For tables with (attribute, value) pairs on the
    `value` field is condidered. For tables with (head, relation, tail) triples a
    concatenation of `head` and `tail` are considered.

    E.g.:
    >>> overlap_probability(["michael", "dahlquist"],
                             [(["name"], ["michael", "dahlquist"])])
    >>> 1.0

    Args:
    ngram: List of tokens.
    table: List of either (attribute, value) pairs or (head, relation, tail)
      triples. Each member of the pair / triple is assumed to already be
      tokenized into a list of strings.
    smoothing: (Optional) Float parameter for laplace smoothing.
    stopwords: (Optional) List of stopwords to ignore (assign P = 1).

    Returns:
    prob: Float probability of ngram being entailed by the table.
    """
    # pylint: disable=g-complex-comprehension
    if len(table[0]) == 2:
        table_values = set([tok for _, value in table for tok in value])
    else:
        table_values = set([tok for head, _, tail in table for tok in head + tail])
    
    overlap = 0
    for token in ngram:
        if stopwords is not None and token in stopwords:
            overlap += 1
            continue
        if token in table_values:
            overlap += 1
    return float(overlap + smoothing) / float(len(ngram) + smoothing)


def _mention_probability(table_entry, sentence, smoothing=0.0):
    """Returns the probability that the table entry is mentioned in the sentence.

    A simple implementation which checks the longest common subsequence between
    the table entry and the sentence. For tables with (attribute, value) pairs
    only the `value` is considered. For tables with (head, relation, tail) triples
    a concatenation of the `head` and `tail` is considered.

    E.g.:
    >>> _mention_probability((["name"], ["michael", "dahlquist"]),
                             ["michael", "dahlquist", "was", "a", "drummer"])
    >>> 1.0

    Args:
    table_entry: Tuple of either (attribute, value) or (head, relation, tail).
      Each member of the tuple is assumed to already be tokenized into a list of
      strings.
    sentence: List of tokens.
    smoothing: Float parameter for laplace smoothing.

    Returns:
    prob: Float probability of entry being in sentence.
    """
    if len(table_entry) == 2:
        value = table_entry[1]
    else:
        value = table_entry[0] + table_entry[2]
    overlap = _len_lcs(value, sentence)
    return float(overlap + smoothing) / float(len(value) + smoothing)


def _len_lcs(x, y):
    """Returns the length of the Longest Common Subsequence between two seqs.

    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    Args:
    x: sequence of words
    y: sequence of words

    Returns
    integer: Length of LCS between x and y
    """
    table = _lcs(x, y)
    n, m = len(x), len(y)
    return table[n, m]


def _lcs(x, y):
    """Computes the length of the LCS between two seqs.

    The implementation below uses a DP programming algorithm and runs
    in O(nm) time where n = len(x) and m = len(y).
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    Args:
    x: collection of words
    y: collection of words

    Returns:
    Table of dictionary of coord and len lcs
    """
    n, m = len(x), len(y)
    table = dict()
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i - 1] == y[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i - 1, j], table[i, j - 1])
    return table


def nwise(iterable, n=2):
    """Yields all ngrams of given order n in iterable."""
    iterables = itertools.tee(iterable, n)
    [next(iterables[i]) for i in range(n) for j in range(i)]
    return zip(*iterables)


def _ngram_counts(sequence, order):
    """Returns count of all ngrams of given order in sequence."""
    if len(sequence) < order:
        return collections.Counter()
    return collections.Counter(nwise(sequence, order))


def parent_instance_level(package,
                          lambda_weight=0.5,
                          smoothing=0.00001,
                          max_order=4,
                          entailment_fn=overlap_probability,
                          mention_fn=_mention_probability):
    
    prediction, reference, table = package  # unpacking
    
    # Weighted ngram precisions and recalls for each order.
    ngram_prec, ngram_rec = list(), list()
    for order in range(1, max_order + 1):
        # Collect n-grams and their entailment probabilities.
        pred_ngram_counts = _ngram_counts(prediction, order)
        pred_ngram_weights = {ngram: entailment_fn(ngram, table)
                              for ngram in pred_ngram_counts}
        ref_ngram_counts = _ngram_counts(reference, order)
        ref_ngram_weights = {ngram: entailment_fn(ngram, table)
                             for ngram in ref_ngram_counts}

        # Precision.
        numerator, denominator = 0., 0.
        for ngram, count in pred_ngram_counts.items():
            denominator += count
            prob_ngram_in_ref = min(
                1., float(ref_ngram_counts.get(ngram, 0) / count))
            numerator += count * (
                prob_ngram_in_ref +
                (1. - prob_ngram_in_ref) * pred_ngram_weights[ngram])
        if denominator == 0.:
            # Set precision to 0.
            ngram_prec.append(0.0)
        else:
            ngram_prec.append(numerator / denominator)

        # Recall.
        numerator, denominator = 0., 0.
        for ngram, count in ref_ngram_counts.items():
            prob_ngram_in_pred = min(
                1., float(pred_ngram_counts.get(ngram, 0) / count))
            denominator += count * ref_ngram_weights[ngram]
            numerator += count * ref_ngram_weights[ngram] * prob_ngram_in_pred
        if denominator == 0.:
            # Set recall to 1.
            ngram_rec.append(1.0)
        else:
            ngram_rec.append(numerator / denominator)

    # Compute recall against table fields.
    table_mention_probs = [mention_fn(entry, prediction)
                     for entry in table]
    table_rec = sum(table_mention_probs) / len(table)

    # Smoothing.
    for order in range(1, max_order):
        if ngram_prec[order] == 0.:
            ngram_prec[order] = smoothing
        if ngram_rec[order] == 0.:
            ngram_rec[order] = smoothing

    # Compute geometric averages of precision and recall for all orders.
    w = 1. / max_order
    if any(prec == 0. for prec in ngram_prec):
        c_prec = 0
    else:
        sp = (w * math.log(p_i) for p_i in ngram_prec)
        c_prec = math.exp(math.fsum(sp))
    if any(rec == 0. for rec in ngram_rec):
        ref_rec = smoothing
    else:
        sr = [w * math.log(r_i) for r_i in ngram_rec]
        ref_rec = math.exp(math.fsum(sr))

    # Combine reference and table recalls.
    if table_rec == 0.:
        table_rec = smoothing
    if ref_rec == 0. or table_rec == 0.:
        c_rec = 0
    else:
        if lambda_weight is None:
            lw = sum([mention_fn(entry, reference) for entry in table
           ]) / len(table)
            lw = 1. - lw
        else:
            lw = lambda_weight

        c_rec = math.exp((1. - lw) * math.log(ref_rec) + (lw) * math.log(table_rec))

    # F-score.
    c_f = (2. * c_prec * c_rec) / (c_prec + c_rec + 1e-8)
    
    return c_prec, c_rec, c_f


def parent(predictions,
           references,
           tables,
           lambda_weight=0.5,
           smoothing=0.00001,
           max_order=4,
           entailment_fn=overlap_probability,
           mention_fn=_mention_probability):
    """Metric for comparing predictions to references given tables.

    Args:
    predictions: An iterator over tokenized predictions.
      Each prediction is a list.
    references: An iterator over lists of tokenized references.
      Each prediction can have multiple references.
    tables: An iterator over the tables. Each table is a list of tuples, where a
      tuple can either be (attribute, value) pair or (head, relation, tail)
      triple. The members of the tuples are assumed to be themselves tokenized
      lists of strings. E.g.
      `[(["name"], ["michael", "dahlquist"]),
      (["birth", "date"], ["december", "22", "1965"])]`
      is one table in the (attribute, value) format with two entries.
    lambda_weight: Float weight in [0, 1] to multiply table recall.
    smoothing: Float value for replace zero values of precision and recall.
    max_order: Maximum order of the ngrams to use.
    entailment_fn: A python function for computing the probability that an
      ngram is entailed by the table. Its signature should match that of
      `overlap_probability` above.
    mention_fn: A python function for computing the probability that a
      table entry is mentioned in the text. Its signature should
        match that of `_mention_probability` above.

    Returns:
    precision: Average precision of all predictions.
    recall: Average recall of all predictions.
    f1: Average F-scores of all predictions.
    all_f_scores: List of all F-scores for each item.
    """
    precisions, recalls, all_f_scores = [], [], []
    
    _parent = partial(parent_instance_level, 
                      lambda_weight=lambda_weight,
                      smoothing=smoothing,
                      max_order=max_order,
                      entailment_fn=entailment_fn,
                      mention_fn=mention_fn)
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        processed_packages = pool.map(_parent, zip(predictions, references, tables))
    
    
    precisions, recalls, all_f_scores = list(), list(), list()
    for p, r, f in processed_packages:
        precisions.append(p)
        recalls.append(r)
        all_f_scores.append(f)

    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_f_score = sum(all_f_scores) / len(all_f_scores)

    return avg_precision, avg_recall, avg_f_score, all_f_scores