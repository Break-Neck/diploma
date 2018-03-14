#!/usr/bin/env python3

import sys
import itertools
import collections


def get_good_words(good_words_file_path):
    with open(good_words_file_path) as fl:
        return set(line.split()[0] for line in fl)


def main():
    if len(sys.argv) != 2 or sys.argv[1] == '-h':
        print('Need 1 argument: path to file with good words')
        sys.exit(1)

    good_words = get_good_words(sys.argv[1])
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
        if current_date != date:
            emit()
            current_date = date
            current_counts.clear()
        current_counts.update(bgram_iterator(words))
    emit()


if __name__ == '__main__':
    main()
