#!/usr/bin/env python3

import sys
import argparse
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser(
        description='Generate and save count matrix')
    parser.add_argument('words_path', help='Path to file with good words. ')
    parser.add_argument(
        'counts_path', help='Path to file with counted wobject.')
    parser.add_argument(
        'output_path', help='Path to file to save matrix into. ')
    parser.add_argument('--tf-idf', help='Calculate tf-idf matrix (word file need to contain document frequencies in 3rd column)',
                        action='store_true', dest='tf_idf')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    with open(args.counts_path) as fl:
        total_lines = sum(1 for _ in fl)
    print('Lines:', total_lines)
    with open(args.words_path) as fl:
        words_indexes = {line.split()[0]: i for i, line in enumerate(fl)}
        if args.tf_idf:
            fl.seek(0)
            document_frequencies = np.array(
                [int(line.strip().split()[2]) for line in fl])
    print('Word-like objects: ', len(words_indexes))
    print('Shape: ', total_lines * len(words_indexes))
    result_array = np.zeros((total_lines, len(words_indexes)))
    with open(args.counts_path) as fl:
        for i, line in enumerate(fl):
            words_counts = line.split()[1:]
            for word, count_str in (tuple(pair.split(':')) for pair in words_counts):
                result_array[i][words_indexes[word]] = float(count_str)
    if args.tf_idf:
        df_mul = np.log(total_lines / document_frequencies)
        for i in range(result_array.shape[0]):
            result_array[i] = np.multiply(result_array[i], df_mul)
    print('Nonzero: ', np.count_nonzero(result_array) / result_array.size)
    np.save(args.output_path, result_array)


if __name__ == '__main__':
    main()
