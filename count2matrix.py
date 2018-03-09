#!/usr/bin/env python3

import sys
import numpy as np


def main():
    if len(sys.argv) != 4:
        print('Usage: ./count2matrix.py good_words_file day_count_file output_file')
        sys.exit(1)
    words_file, input_file, output_file = tuple(sys.argv[1:])
    with open(input_file) as fl:
        total_lines = sum(1 for _ in fl)
    print('Lines:', total_lines)
    with open(words_file) as fl:
        words_indexes = {line.split()[0]: i for i, line in enumerate(fl)}
    print('Word-like objects: ', len(words_indexes))
    result_array = np.zeros((total_lines, len(words_indexes)))
    with open(input_file) as fl:
        for i, line in enumerate(fl):
            words_counts = line.split()[1:]
            for word, count_str in (tuple(pair.split(':')) for pair in words_counts):
                result_array[i][words_indexes[word]] = int(count_str)
    print('Nonzero: ', np.count_nonzero(result_array) / result_array.size)
    np.save(output_file, result_array)


if __name__ == '__main__':
    main()
