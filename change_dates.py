#!/usr/bin/env python3

import sys
import csv
import dateutil.parser
import datetime


def read_course_dates(file_path):
    coursed_dates = []
    with open(file_path) as fl:
        next(fl)
        reader = csv.reader(fl)
        for row in reader:
            date = dateutil.parser.parse(row[0], dayfirst=True).date()
            coursed_dates.append(date)
    assert len(coursed_dates) == len(set(coursed_dates))
    return set(coursed_dates)


class CourseDateFinder():
    step_time = datetime.time(8, 30)

    def __init__(self, coursed_dates):
        self.coursed_dates = coursed_dates
        self.min_date, self.max_date = min(
            self.coursed_dates), max(self.coursed_dates)

    def _find_known_date(self, date, days_step):
        while self.min_date <= date <= self.max_date and date not in self.coursed_dates:
            date += datetime.timedelta(days=days_step)
        if not (self.min_date <= date <= self.max_date):
            raise ValueError('Date out of range')
        return date

    def find(self, date_with_time):
        if date_with_time.time() < self.step_time:
            start_day = date_with_time.date() + datetime.timedelta(days=1)
        else:
            start_day = date_with_time.date() + datetime.timedelta(days=2)
        return self._find_known_date(start_day, 1)


def main():
    if len(sys.argv) < 2:
        print('One argument needed: path to a course file')
        sys.exit(1)
    coursed_dates = read_course_dates(sys.argv[1])
    course_finder = CourseDateFinder(coursed_dates)

    for line in sys.stdin:
        line = line.strip()
        if line.count(' ') < 2:
            continue
        date_time_end = line.find(' ', line.find(' ') + 1)
        new_date = course_finder.find(
            dateutil.parser.parse(line[:date_time_end]))
        print(str(new_date) + line[date_time_end:])


if __name__ == '__main__':
    main()
