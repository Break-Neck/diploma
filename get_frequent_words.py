#!/usr/bin/env python3

import sys
import collections
import itertools
import argparse

def count_from_text(input, skip_start):
    counter = collections.Counter()
    for line in input:
        lemmas = line.split()[skip_start:]
        counter.update(itertools.chain(lemmas, ('|'.join(pair) for pair in zip(lemmas[:-1], lemmas[1:]))))
        counter.update(line.split()[skip_start:])
    return counter


def merge_counters(input):
    count_dict = dict()
    for line in input:
        wobj, wobj_count_str = tuple(line.split())
        count_dict[wobj] = count_dict.get(wobj, 0) + int(wobj_count_str)
    return count_dict


def get_parser():
    parser = argparse.ArgumentParser(description='Count words in text or merge counts and print only frequent ones. ')
    parser.add_argument('-s', '--skip', help='Number of fields to skip from beginning of the line. ', type=int)
    parser.add_argument('-f', '--freq', help='Minimal frequency for a word or bigram to be printed. ', type=int)
    parser.add_argument('-m', '--merge', help='Turning into merge mode. ', action='store_true')
    return parser


def main():
    args = get_parser().parse_args()
    if args.merge:
        final_data = merge_counters(sys.stdin)
    else:
        final_data = count_from_text(sys.stdin, args.skip)
    for wobj, count in final_data.items():
        if count >= args.freq:
            print('{} {}'.format(wobj, count))


if __name__ == '__main__':
    main()

