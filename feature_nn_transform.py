import numpy as np
import dateutil
import itertools


def iterate_bgrams(lemmas_list):
    return itertools.chain(lemmas_list, ('|'.join(a, b) for a, b in zip(lemmas_list[:-1], lemmas_list[1:])))


def date_from_course_string(date_str):
    return dateutil.parser.parse(date_str, day_first=True).date()


class FeatureTfIdfTransformer(object):
    __up_y = np.array([1., 0.])
    __down_y = np.array([0., 1.])
    __stagnate_y = np.array([.5, .5])

    def __init__(self, data_file_path, good_words_path, course_file_path, validate_size, total_document_count=None, standartize=True):
        self.data_file_path = data_file_path
        self.word2index = dict()
        if total_document_count:
            self.total_document_count = total_document_count
        else:
            with open(data_file_path) as fl:
                self.total_document_count = sum(1 for _ in fl) - validate_size
        self.index2idf_array = None
        self._init_words_counts(good_words_path)
        self.dates2change = self._get_courses(course_file_path)
        self.validate_size = validate_size
        self.standartize_std = 0
        self.standartize_mean = 1
        if standartize:
            self.calculate_standartize()

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
        self.standartize_std = 0
        self.standartize_mean = 1
        sum_by_whole_data, sum_of_squares = None, None
        for row_array in itertools.islice(self.iterate_train_data(1), 0, self.total_document_count):
            if sum_by_whole_data is not None:
                sum_by_whole_data += row_array[0]
            else:
                sum_by_whole_data = row_array[0].copy()
            if sum_of_squares is not None:
                sum_of_squares += row_array[0] ** 2
            else:
                sum_of_squares = row_array[0] ** 2
        self.standartize_mean = sum_by_whole_data / self.total_document_count
        self.standartize_std = 1 / self.total_document_count * sum_of_squares - sum_by_whole_data ** 2
        # correctate std
        self.standartize_std *= np.sqrt(self.total_document_count / (self.total_document_count - 1))

    @staticmethod
    def _get_courses(file_path):
        dates2change = dict()
        with open(file_path) as fl:
            for line in fl:
                splitted = line.strip().split(',')
                date = date_from_course_string(splitted[0])
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
        X, y = np.zeros((len(data), len(self.word2index))), np.zeros(len(data), 2)
        for i, record in enumerate(data):
            X[i, ...] = self._line2count_vector(record[1:])
            true_y = np.sign(self.dates2change[date_from_course_string(record[0])])
            if true_y > .5:
                y[i, ...] = self.__up_y
            elif true_y < -.5:
                y[i, ...] = self.__down_y
            else:
                y[i, ...] = self.__stagnate_y
        return X, y

    def apply_tfidf(self, X):
        for i in range(X.shape[0]):
            X[i, ...] = np.multiply(X[i, ...], self.index2idf_array)

    def standartize(self, X):
        for i in range(X.shape[0]):
            X[i, ...] = (X[i, ...] - self.standartize_mean) / self.standartize_std

    def _iterate_processed_data(self, data_iterator, batch_size):
        while True:
            data_chunk = list(itertools.islice(data_iterator, 0, batch_size))
            X, y = self.count_vectorize(data_chunk)
            self.apply_tfidf(X)
            self.standartize(X)
            return X, y

    def iterate_train_data(self, batch_size):
        data_iterator = self.iterate_data(self.validate_size)
        return self._iterate_processed_data(data_iterator, batch_size)

    def iterate_validation_data(self, batch_size):
        data_iterator = self.iterate_data(0, self.validate_size)
        return self._iterate_processed_data(data_iterator, batch_size)
