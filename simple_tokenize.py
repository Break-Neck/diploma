#!/usr/bin/env python3

import jsonlines
import sys
import string


def tokenize(text):
    chars_to_prune = set(string.punctuation) - {"'"}
    text = ''.join(ch for ch in text.strip() if ch not in chars_to_prune)
    split = [s.lower() for s in text.split()]
    with_not = []
    had_not_before = 0
    for tok in split:
        if tok.endswith("n't"):
            with_not.append(tok[:-3])
            had_not_before = 2
        elif tok == 'not':
            had_not_before = 2
        elif tok.isalnum():
            if had_not_before > 0:
                with_not.append('not_' + tok)
                had_not_before -= 1
            else:
                with_not.append(tok)
    return with_not


if __name__ == '__main__':
    for record in jsonlines.Reader(sys.stdin):
        print(' '.join([record['date']] + tokenize(record['text'])))
