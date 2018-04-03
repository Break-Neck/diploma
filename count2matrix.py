#!/usr/bin/env python3

import sys
import argparse
import numpy as np
import scipy.sparse
import gc


def get_parser():
    parser = argparse.ArgumentParser(
        description='Generate and save count matrix')
    parser.add_argument('words_path', help='Path to file with good words. ')
    parser.add_argument(
        'counts_path', help='Path to file with counted wobject.')
    parser.add_argument(
        'output_path', help='Path to file to save matrix into. ')
    parser.add_argument('-s', help='Produce sparse matrix', action='store_true', dest='sparse')
    return parser


class NumPyMatrixConsumer:
    def __init__(self, shape):
        self.shape = shape
        self.matrix = np.zeros(shape)
        self.current_row = 0

    def consume(self, wobj_index, value):
        self.matrix[current_row][wobj_index] = value

    def new_line(self):
        self.current_row += 1

    def produce(self):
        return self.matrix

class SparseConsumer:
    def __init__(self, shape):
        self.shape = shape
        self.coo_data = []
        self.coo_rows = []
        self.coo_columns = []
        self.current_row = 0

    def consume(self, wobj_index, value):
        self.coo_data.append(value)
        self.coo_rows.append(self.current_row)
        self.coo_columns.append(wobj_index)

    def new_line(self):
        self.current_row += 1
        if self.current_row % 1000 == 0:
            gc.collect()

    def produce(self):
        return scipy.sparse.coo_matrix(self.coo_data, (self.coo_rows, self.coo_columns)).tocsr()

def main():
    parser = get_parser()
    args = parser.parse_args()
    with open(args.counts_path) as fl:
        total_lines = sum(1 for _ in fl)
    print('Lines:', total_lines)
    with open(args.words_path) as fl:
        words_indexes = {line.split()[0]: i for i, line in enumerate(fl)}
    print('Word-like objects: ', len(words_indexes))
    print('Total size: ', total_lines * len(words_indexes))
    if args.sparse:
        consumer = SparseConsumer((total_lines, len(words_indexes)))
    else:
        consumer = NumPyMatrixConsumer((total_lines, len(words_indexes)))
    with open(args.counts_path) as fl:
        for line in fl:
            words_counts = line.split()[1:]
            for word, count_str in (tuple(pair.split(':')) for pair in words_counts):
                consumer.consume(words_indexes[word], float(count_str))
            consumer.new_line()
    result_array = consumer.produce()
    if args.sparse:
        print('Nonzero: ', result_array.count_nonzero())
        scipy.sparse.save_npz(args.output_path, result_array)
    else:
        print('Nonzero: ', np.count_nonzero(result_array) / result_array.size)
        np.save(args.output_path, result_array)


if __name__ == '__main__':
    main()
