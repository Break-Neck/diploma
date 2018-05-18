#!/usr/bin/env python3

import sys
import argparse

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('answer_file', help='Path to answer file')
    return parser


def main():
    parser = get_arg_parser()
    args = parser.parse_args()

    with open(args.answer_file) as fl:
        date_target = dict(line.strip().split(',') for line in fl)

    for line in sys.stdin:
        line = line.strip().replace('|', '_')
        date, values = tuple(line.split(' ', 1))
        if date in date_target:
            print('{} {}| '.format(date_target[date], date), end='')
            print(values)

if __name__ == '__main__':
    main()
