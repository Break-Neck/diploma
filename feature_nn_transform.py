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

class FeatureTfIdfTransformer(object):
    __up_y = np.array([1., 0.])
    __down_y = np.array([0., 1.])
    __stagnate_y = np.array([.5, .5])

    def __init__(self, data_file_path, good_words_path, course_file_path, train_part, validate_part, total_document_count=None,
                 standartize_path=None):
        self.data_file_path = data_file_path
        self.word2index = dict()
        if total_document_count:
            self.total_document_count = total_document_count
        else:
            logging.info('Start counting documents')
            with open(data_file_path) as fl:
                self.total_document_count = sum(1 for _ in fl)
            logging.info('Finished counting documents')
        self.index2idf_array = None
        logging.info('Start initialization of words statistics')
        self._init_words_counts(good_words_path)
        logging.info('Finished initialization of words statistics')
        logging.debug('Start reading of courses data')
        self.dates2change = self._get_courses(course_file_path)
        logging.debug('Finished initialization of words statistics')
        self.train_size = int(self.total_document_count * train_part)
        self.validate_size = int(self.total_document_count * validate_size)
        self.preprocessing_pipeline = None
        self.standartize_mean = 0
        self.standartize_std = 1
        if standartize_path is None:
            logging.info('Start calculation of data mean/std')
            self.calculate_standartize()
            logging.info('Finished calculation of data mean/std')
        else:
            self.standartize_mean, self.standartize_std = self.load_standartize_stat(standartize_path)

    def _init_words_counts(self, file_path):
        index2document_frequency = []
        with open(file_path) as fl:
            for i, line in enumerate(fl):
                splitted = line.strip().split()
                wobj, document_frequency = splitted[0], int(splitted[2])
                self.word2index[wobj] = i
                index2document_frequency.append(document_frequency)
        index2document_frequency = np.array(index2document_frequency)
        self.index2idf_array = np.log(self.total_document_count / index2document_frequency)

    def calculate_standartize(self):
        sum_by_whole_data, sum_of_squares = None, None
        for i, row_array in enumerate(itertools.islice(self.iterate_train_data(1), 0, self.total_document_count)):
            if sum_by_whole_data is not None:
                sum_by_whole_data += row_array[0, :]
            else:
                sum_by_whole_data = row_array[0, :].copy()
            if sum_of_squares is not None:
                sum_of_squares += row_array[0, :] ** 2
            else:
                sum_of_squares = row_array[0, :] ** 2
            if i % 100000 == 0:
                logging.debug('Processed %f', i / self.total_document_count)
        self.standartize_mean = sum_by_whole_data / self.total_document_count
        self.standartize_std = 1 / self.total_document_count * sum_of_squares - sum_by_whole_data ** 2
        # correct std
        self.standartize_std *= np.sqrt(self.total_document_count / (self.total_document_count - 1))
        self.standartize_mean = np.squeeze(self.standartize_mean)
        self.standartize_std = np.squeeze(self.standartize_std)

    def save_standartize_stat(self, path):
        np.savez(path, mean=self.standartize_mean, std=self.standartize_std)

    @staticmethod
    def load_standartize_stat(path):
        arrays = np.load(path)
        return arrays['mean'], arrays['std']

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

    def _line2count_vector(self, lemmas_in_line):
        line_vector = np.zeros(len(self.word2index))
        for wobj in iterate_bgrams(lemmas_in_line):
            if wobj in self.word2index:
                line_vector[self.word2index[wobj]] += 1
        return line_vector

    def iterate_data(self, start_line_number=0, line_count=None):
        with open(self.data_file_path) as data_fl:
            for _ in range(start_line_number):
                data_fl.readline()
            start_position = data_fl.tell()
            while True:
                for line in itertools.islice(data_fl, 0, line_count):
                    yield line.strip().split()
                data_fl.seek(start_position)

    def count_vectorize(self, data):
        X, y = np.zeros((len(data), len(self.word2index))), np.zeros((len(data), 2))
        for i, record in enumerate(data):
            X[i] = self._line2count_vector(record[1:])
            true_y = np.sign(self.dates2change[date_from_data_file_string(record[0])])
            if true_y > .5:
                y[i] = self.__up_y
            elif true_y < -.5:
                y[i] = self.__down_y
            else:
                y[i] = self.__stagnate_y
        return X, y

    def apply_tfidf(self, X):
        for i in range(X.shape[0]):
            X[i] = np.multiply(X[i], self.index2idf_array)

    def standartize(self, X):
        for i in range(X.shape[0]):
            X[i] = (X[i] - self.standartize_mean) / self.standartize_std

    def _iterate_processed_data(self, data_iterator, batch_size, gc_period=None):
        index = 0
        while True:
            data_chunk = list(itertools.islice(data_iterator, 0, batch_size))
            X, y = self.count_vectorize(data_chunk)
            self.apply_tfidf(X)
            self.standartize(X)
            yield X, y
            index += 1
            if gc_period is not None and index % gc_period == 0:
                logging.debug('GC\'ed')
                gc.collect()

    def iterate_train_data(self, batch_size, gc_period=None):
        data_iterator = self.iterate_data(line_count=self.train_size)
        for batch in self._iterate_processed_data(data_iterator, batch_size, gc_period):
            logging.debug('Train batch generated')
            yield batch

    def iterate_validation_data(self, batch_size, gc_period=None):
        data_iterator = self.iterate_data(0, self.validate_size)
        for batch in self._iterate_processed_data(data_iterator, batch_size, gc_period):
            logging.debug('Validation batch generated')
            yield batch

    @property
    def vector_size(self):
        return len(self.word2index)


class PureIterableProvider:
    def iterate_train_chunk(self):
        return self._iterate_chunk(0, self.train_dates)
    
    def iterate_validation_chunk(self):
        return self._iterate_chunk(self.train_dates, self.validation_dates)

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

    def _iterate_chunk(self, start_date_index, dates_count):
        while True:
            for date_index in range(start_date_index, start_date_index + dates_count):
                news_count = self._get_news_count_for_date(date_index)
                X = self.sparse_matrix[self.dates_lines[date_index]:self.dates_lines[date_index] + news_count].toarray()
                y = self._get_y_by_date(date_index)
                yield np.expand_dims(X, 0), y

class RandomDateOrderIterator(SparseIterator):
    def __init__(self, data_file_path, good_words_path, course_file_path, sparse_matrix_path, train_part, validate_part, dates_load=None, seed=None):
        super().__init__(data_file_path, good_words_path, course_file_path, sparse_matrix_path, train_part, validate_part, dates_load)

    def _iterate_chunk(self, start_date_index, dates_count):
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

    def _iterate_chunk(self, start_date_index, dates_count):
        for X, y in self.sparse_iterator._iterate_chunk(start_date_index, dates_count):
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

    def _iterate_chunk(self, start_date_index, dates_count, scale=False):
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
        for chunk in itertools.islice(self.iterate_train_chunk(False), self.train_dates):
            scaler.partial_fit(np.squeeze(chunk[0]))
        return scaler

    def save_scaler(self, save_path=__DEFAULT_SCALER_PATH):
        with open(save_path, 'wb') as fl:
            pickle.dump(self.scaler, fl)

    def load_scaler(self, load_path=__DEFAULT_SCALER_PATH):
        with open(load_path, 'rb') as fl:
            self.scaler = pickle.load(fl)

