#!/usr/bin/env python3

import sys
import json

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

__tag_convertion_map = {
    'N' : 'n',
    'V' : 'v',
    'J' : 'a',
    'S' : 's',
    'R' : 'r'
}
__wnl = WordNetLemmatizer()


def apply_not(lemmas, apply_length=2):
    has_not_flag = 0
    for lemma in lemmas:
        if lemma == 'not':
            has_not_flag = apply_length
        elif has_not_flag > 0:
            yield 'not_' + lemma
            has_not_flag -= 1
        else:
            yield lemma


def convert_tags(old_tag):
    return __tag_convertion_map.get(old_tag[0], 'n')


def get_lems(text):
    tokens = word_tokenize(text)
    for i, tok in enumerate(tokens):
        if tok.endswith("n't"):
            tokens[i] = 'not'
    tagged = pos_tag([tok.lower() for tok in tokens if tok.isalnum()])
    return apply_not(__wnl.lemmatize(tok, convert_tags(tag)) for tok, tag in tagged)


if __name__ == '__main__':
    for line in sys.stdin:
        json_data = json.loads(line)
        date_str = json_data['date']
        text = json_data['text']
        print('{} {}'.format(date_str, ' '.join(get_lems(text))))

