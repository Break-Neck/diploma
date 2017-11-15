#!/usr/bin/env python3

import sys

def main():
    last_line, last_count = None, 0

    def emit():
        print('{},{}'.format(last_line, last_count))

    for line in sys.stdin:
        current_line, current_count_str = tuple(line.strip().rsplit(',', 1))
        if current_line != last_line:
            if last_line:
                emit()
            last_line = current_line
            last_count = int(current_count_str)
        else:
            last_count += int(current_count_str)
    if last_line:
        emit()



if __name__ == '__main__':
    main()

