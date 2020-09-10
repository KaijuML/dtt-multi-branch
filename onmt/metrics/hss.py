import json

import numpy as np

from data.co_occurrence import build_scores, build_sentence_object, fuse_pos_and_deprel, interesting_tags, avg_expand, \
    handle_sentence_punctuation


def _load_stanza():
    """
    Bi-LSTM-based deep biaffine neural dependency parser (Dozat and Manning, 2017), augmented with two linguistically
    motivated features: one that predicts the linearization order of two words in a given language, and the other that
    predicts the typical distance in linear order between them.
    UD English EWT
    UAS 86.22%
    LAS 83.59%
    """
    import stanza
    try:
        model = stanza.Pipeline(
            lang="en", processors='tokenize,pos,lemma,depparse')
    except FileNotFoundError:
        stanza.download('en')
        model = stanza.Pipeline(
            lang="en", processors='tokenize,pos,lemma,depparse')
    model.processors['tokenize'].config['pretokenized'] = True
    return model


class _HallucinationScore:
    def __init__(self, co_occur_file, tables_file):
        with open(co_occur_file) as f:
            self.co_occur = json.load(f)
        with open(tables_file) as f:
            self.inputs = [json.loads(line) for line in f]
        self.nlp = _load_stanza()

    def __call__(self, sentence, i):
        if len(sentence) == 0:
            return [0]
        input_table = self.inputs[i]
        sentence = self.nlp([sentence]).sentences[0]
        pos = [(token.text, token.upos) for token in sentence.words]
        deprel = [(token.text, token.deprel, token.head) for token in sentence.words]

        scores = build_scores(input_table, self.co_occur)
        sentence = build_sentence_object(fuse_pos_and_deprel(pos, deprel))

        h = [float(token.pos_ in interesting_tags) for token in sentence]

        # Score interesting tokens
        for token in sentence:
            if token.pos_ in interesting_tags:
                h[token.i] -= scores.get(token.text, 0)

        # Expand the hallucination scores according to the chosen strategy
        avg_expand(sentence, h)

        # Some tricks to harmonize the scoring
        handle_sentence_punctuation(sentence, h)
        return h


class HSS(_HallucinationScore):
    def __call__(self, sentence, i):
        return sum(super().__call__(sentence, i))


class HSA(_HallucinationScore):
    def __call__(self, sentence, i):
        return np.mean(super().__call__(sentence, i))
