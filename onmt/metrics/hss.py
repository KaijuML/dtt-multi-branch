import json

from data.co_occurrence import build_scores, build_sentence_object, fuse_pos_and_deprel, interesting_tags, avg_expand, \
    handle_sentence_punctuation
from data.dep_parsing_all import _load_stanza


class HSS:
    def __init__(self, co_occur_file, tables_file):
        with open(co_occur_file) as f:
            self.co_occur = json.load(f)
        with open(tables_file) as f:
            self.inputs = [json.loads(line) for line in f]
        self.nlp = _load_stanza()

    def __call__(self, sentence, i):
        input_table = self.inputs[i]
        parsed = self.nlp(sentence)
        input_pos, input_deprel = None, None  # TODO from 'parsed' object
        breakpoint()

        scores = build_scores(input_table, self.co_occur)
        sentence = build_sentence_object(fuse_pos_and_deprel(input_pos, input_deprel))

        h = [float(token.pos_ in interesting_tags) for token in sentence]

        # Score interesting tokens
        for token in sentence:
            if token.pos_ in interesting_tags:
                h[token.i] -= scores.get(token.text, 0)

        # Expand the hallucination scores according to the chosen strategy
        avg_expand(sentence, h)

        # Some tricks to harmonize the scoring
        handle_sentence_punctuation(sentence, h)
        return sum(h)
