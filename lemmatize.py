#!/usr/bin/env python3

import sys
import json

from nltk.tokenize import casual_tokenize
from nltk.stem import WordNetLemmatizer


def apply_not(lemmas, apply_length=2):
    new_lemmas = []
    has_not_flag = 0
    for lemma in lemmas:
        if lemma == 'not':
            has_not_flag = apply_length
        elif has_not_flag > 0:
            new_lemmas.append('not_' + lemma)
            has_not_flag -= 1
        else:
            new_lemmas.append(lemma)
    return new_lemmas


def get_lems(text):
    tokens = casual_tokenize(text, preserve_case=False)
    for i, tok in enumerate(tokens):
        if tok.endswith("n't"):
            tokens[i] = 'not'
    wnl = WordNetLemmatizer()

    def dummy_lemmatize(token):
        n_lemma = wnl.lemmatize(token, 'n')
        v_lemma = wnl.lemmatize(token, 'v')
        return n_lemma if len(n_lemma) <= len(v_lemma) else v_lemma

    lemmas = [dummy_lemmatize(tok) for tok in tokens if tok.isalnum()]
    return apply_not(lemmas)


if __name__ == '__main__':
    for line in sys.stdin:
        json_data = json.loads(line)
        date_str = json_data['date']
        text = json_data['text']
        print('{} {}'.format(date_str, ' '.join(get_lems(text))))
