#!/usr/bin/env python3

import argparse
import sys
import itertools
import collections


def get_good_words(good_words_file_path):
    with open(good_words_file_path) as fl:
        return set(line.split()[0] for line in fl)


def get_parser():
    parser = argparse.ArgumentParser(description='Leave only important lemmas/word-like object. ')
    parser.add_argument('words_path', help='Path to file with good words. ')
    parser.add_argument('-p', help='Don\'t compress records with same dates into one', action='store_false', dest='compress')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    good_words = get_good_words(args.words_path)
    current_date = None
    current_counts = collections.Counter()

    def bgram_iterator(words):
        return (x for x in itertools.chain(words, ('|'.join(y) for y in zip(words[:-1], words[1:]))) if x in good_words)

    def emit():
        if current_date is not None:
            print(' '.join([current_date] + ['{}:{}'.format(word, count)
                                             for word, count in current_counts.items()]))

    for line in sys.stdin:
        split = line.strip().split()
        date, words = split[0], split[1:]
        if not args.compress or current_date != date:
            emit()
            current_date = date
            current_counts.clear()
        current_counts.update(bgram_iterator(words))
    emit()


if __name__ == '__main__':
    main()
