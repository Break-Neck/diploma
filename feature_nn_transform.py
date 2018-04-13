import numpy as np
import datetime
import itertools
import os
import sys
import logging
import pickle
import sklearn
import sklearn.decomposition
import sklearn.preprocessing
import scipy
import scipy.sparse
import gc


def date_from_course_file_string(date_str):
    return datetime.datetime.strptime(date_str, '%d.%m.%Y').date()

def date_from_data_file_string(date_str):
    return datetime.datetime.strptime(date_str, '%Y-%m-%d').date()

def iterate_wobj_counts(wobj_data):
    for wdata in wobj_data:
        wobj, wobj_count = tuple(wdata.split(':'))
        yield wobj, float(wobj_count)

class PureDateIterableProvider:
    def iterate_train_dates_chunk(self):
        return self._iterate_dates_chunk(0, self.train_dates)
    
    def iterate_validation_dates_chunk(self):
        return self._iterate_dates_chunk(self.train_dates, self.validation_dates)

class BaseDataVectorizer(PureIterableProvider):
    __DEFAULT_DATES_INFO_PATH = 'dates_info.pl'

    __up_y = np.array([1])
    __down_y = np.array([0])

    def __init__(self, data_file_path, good_words_path, course_file_path, train_part, validate_part, dates_load=None):
        self.data_file_path = data_file_path
        self.wobj2num = self._get_good_words_encoding(good_words_path)
        self.dates2change = self._get_courses(course_file_path)
        if dates_load is None:
            logging.info('Start dates info retrieving')
            self.all_dates, self.dates_lines, self.total_lines = self._get_dates_info()
            logging.info('Dates info received')
        else:
            load_path = dates_load if isinstance(dates_load, str) else self.__DEFAULT_DATES_INFO_PATH
            self.load_dates_info(load_path)
        self.train_dates = int(len(self.all_dates) * train_part)
        self.validation_dates = int(len(self.all_dates) * validate_part)

    @property
    def vector_length(self):
        return len(self.wobj2num)

    @staticmethod
    def _get_good_words_encoding(good_words_path):
        with open(good_words_path) as fl:
            return {line.split()[0]:i for i, line in enumerate(fl)}

    @staticmethod
    def _get_courses(file_path):
        dates2change = dict()
        with open(file_path) as fl:
            next(fl)
            for line in fl:
                splitted = line.strip().split(',')
                date = date_from_course_file_string(splitted[0])
                absolute_change = float(splitted[2])
                dates2change[date] = absolute_change
        return dates2change

    def _get_dates_info(self):
        dates = set()
        dates_lines = []
        line_number = 0
        with open(self.data_file_path) as fl:
            for line in fl:
                date = date_from_data_file_string(line.split()[0])
                if date not in dates:
                    dates_lines.append(line_number)
                    dates.add(date)
                line_number += 1
        return sorted(dates), dates_lines, line_number

    def save_dates_info(self, save_path=__DEFAULT_DATES_INFO_PATH):
        with open(save_path, 'wb') as fl:
            pickle.dump((self.all_dates, self.dates_lines, self.total_lines), fl)

    def load_dates_info(self, load_path=__DEFAULT_DATES_INFO_PATH):
        with open(load_path, 'rb') as fl:
            self.all_dates, self.dates_lines, self.total_lines = pickle.load(fl)

    def _get_news_count_for_date(self, date_index):
        if date_index + 1 == len(self.all_dates):
            news_count = self.total_lines - self.dates_lines[date_index]
        else:
            news_count = self.dates_lines[date_index + 1] - self.dates_lines[date_index]
        return news_count
    
    def _get_y_by_date(self, date_index):
        if self.dates2change[self.all_dates[date_index]] < 0:
            return self.__down_y
        else:
            return self.__up_y

class SparseIterator(BaseDataVectorizer):
    def __init__(self, data_file_path, good_words_path, course_file_path, sparse_matrix_path, train_part, validate_part, dates_load=None):
        super().__init__(data_file_path, good_words_path, course_file_path, train_part, validate_part, dates_load)
        self.sparse_matrix = scipy.sparse.load_npz(sparse_matrix_path)

    def _iterate_dates_chunk(self, start_date_index, dates_count):
        while True:
            for date_index in range(start_date_index, start_date_index + dates_count):
                news_count = self._get_news_count_for_date(date_index)
                X = self.sparse_matrix[self.dates_lines[date_index]:self.dates_lines[date_index] + news_count].toarray()
                y = self._get_y_by_date(date_index)
                yield np.expand_dims(X, 0), y

class RandomDateOrderIterator(SparseIterator):
    def __init__(self, data_file_path, good_words_path, course_file_path, sparse_matrix_path, train_part, validate_part, dates_load=None, seed=None):
        super().__init__(data_file_path, good_words_path, course_file_path, sparse_matrix_path, train_part, validate_part, dates_load)

    def _iterate_dates_chunk(self, start_date_index, dates_count):
        iteration_dates_indexes = np.arange(start_date_index, start_date_index + dates_count)
        while True:
            for date_index in iteration_dates_indexes:
                news_count = self._get_news_count_for_date(date_index)
                X = self.sparse_matrix[self.dates_lines[date_index]:self.dates_lines[date_index] + news_count].toarray()
                y = self._get_y_by_date(date_index)
                yield np.expand_dims(X, 0), y
            np.random.shuffle(iteration_dates_indexes)

class ScaleIteratorWrapper(PureIterableProvider):
    __DEFAULT_SCALER_PATH = 'nn_scaler.pl'

    def __init__(self, sparse_iterator, scaler_load=None, pretransform=False):
        self.sparse_iterator = sparse_iterator
        self.train_dates = self.sparse_iterator.train_dates
        self.validation_dates = self.sparse_iterator.validation_dates
        self.pretransform = pretransform
        self._init_scaler(scaler_load, pretransform)

    def save_scaler(self, save_path=__DEFAULT_SCALER_PATH):
        with open(save_path, 'wb') as fl:
            pickle.dump(self.scaler, fl)

    def load_scaler(self, load_path=__DEFAULT_SCALER_PATH):
        with open(load_path, 'rb') as fl:
            self.scaler = pickle.load(fl)

    def _init_scaler(self, scaler_load, pretransform):
        if scaler_load is None:
            self.scaler = None
            return
        if hasattr(scaler_load, 'transform'):
            self.scaler = scaler_load
            self.scaler.fit(self.sparse_iterator.sparse_matrix)
        elif scaler_load:
            load_path = scaler_load if isinstance(scaler_load, str) else self.__DEFAULT_SCALER_PATH
            self.load_scaler(load_path)
        if pretransform:
            self.sparse_iterator.sparse_matrix = self.scaler.transform(self.sparse_iterator.sparse_matrix, copy=False)

    def _iterate_dates_chunk(self, start_date_index, dates_count):
        for X, y in self.sparse_iterator._iterate_dates_chunk(start_date_index, dates_count):
            if not self.pretransform:
                X = np.expand_dims(self.scaler.transform(np.squeeze(X), copy=False), 0)
            yield X, y


class CountDataVectorizer(BaseDataVectorizer):
    __DEFAULT_SCALER_PATH = 'nn_scaler.pl'

    def __init__(self, data_file_path, good_words_path, course_file_path, train_part, validate_part, dates_load=None, scaler_load=None):
        super().__init__(data_file_path, good_words_path, course_file_path, train_part, validate_part, dates_load)
        if scaler_load is None:
            logging.info('Start scaling calculation')
            self.scaler = self._get_scaler()
            logging.info('Finished scaling calculation')
        else:
            load_path = scaler_load if isinstance(scaler_load, str) else self.__DEFAULT_SCALER_PATH
            self.load_scaler(load_path)

    def _read_n_records(self, file_object, n_lines):
        X = np.zeros((n_lines, self.vector_length))
        dates = set()
        for i, line in itertools.islice(enumerate(file_object), n_lines):
            splitted = line.split()
            dates.add(splitted[0])
            for wobj, wobj_count in iterate_wobj_counts(splitted[1:]):
                X[i][self.wobj2num[wobj]] = wobj_count
        assert len(dates) == 1
        return X

    def _iterate_dates_chunk(self, start_date_index, dates_count, scale=False):
        with open(self.data_file_path) as fl:
            for _ in range(self.dates_lines[start_date_index]):
                fl.readline()
            start_seek_position = fl.tell()
            while True:
                for date_index in range(start_date_index, start_date_index + dates_count):
                    news_count = self._get_news_count_for_date(date_index)
                    X = self._read_n_records(fl, news_count)
                    y = self._get_y_by_date(date_index)
                    if scale:
                        self.scaler.transform(X)
                    yield np.expand_dims(X, 0), np.expand_dims(y, 0)
                fl.seek(start_seek_position)

    def _get_scaler(self):
        scaler = sklearn.preprocessing.StandardScaler()
        for chunk in itertools.islice(self.iterate_train_dates_chunk(False), self.train_dates):
            scaler.partial_fit(np.squeeze(chunk[0]))
        return scaler

    def save_scaler(self, save_path=__DEFAULT_SCALER_PATH):
        with open(save_path, 'wb') as fl:
            pickle.dump(self.scaler, fl)

    def load_scaler(self, load_path=__DEFAULT_SCALER_PATH):
        with open(load_path, 'rb') as fl:
            self.scaler = pickle.load(fl)

