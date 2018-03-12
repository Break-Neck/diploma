#!/usr/bin/env python3

import sys

def main():
    last_line, last_count = None, 0

    def emit():
        if last_line:
            print('{}|{}'.format(last_line, last_count))

    for line in sys.stdin:
        current_line = line.strip().rsplit('|', 1)[-1]
        if current_line != last_line:
            emit()
            last_line = current_line
            last_count = 1
        else:
            last_count += 1
    emit()



if __name__ == '__main__':
    main()

