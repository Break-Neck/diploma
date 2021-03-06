#!/usr/bin/env python3

import numpy as np
import sys
import logging
import argparse
import sklearn.preprocessing
import feature_nn_transform


logger = logging.getLogger()
logger.setLevel(logging.INFO)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help='Model name for files. ', default='model')
    parser.add_argument('--lstm', help='Neurons in lstm. ', type=int, default=16)
    return parser

parser = get_parser()
args = parser.parse_args()

import keras

train_part = 0.8
validate_part = 0.15
data_processor = feature_nn_transform.RandomDateOrderIterator('lemmas/ccl.txt', 'lemmas/wc.txt', 'course.csv', 'lemmas/smd.npz',
        train_part, validate_part, dates_load=True)
normalized_iterator = feature_nn_transform.ScaleIteratorWrapper(data_processor, scaler_load=sklearn.preprocessing.Normalizer(copy=False))

model = keras.models.Sequential()
model.add(keras.layers.GaussianDropout(0.2, input_shape=(None, data_processor.vector_length)))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(1000, activation='relu')))
model.add(keras.layers.GaussianDropout(0.2))
model.add(keras.layers.LSTM(args.lstm))
model.add(keras.layers.GaussianDropout(0.2))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['acc'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint(filepath='nn/{}.hdf5'.format(args.name), save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.25, patience=3, verbose=1, cooldown=3),
    keras.callbacks.CSVLogger('nn/{}.csv'.format(args.name)),
    keras.callbacks.EarlyStopping(patience=8)
]

model.fit_generator(
    normalized_iterator.iterate_train_chunk(),
    normalized_iterator.train_dates,
    validation_data=normalized_iterator.iterate_validation_chunk(),
    validation_steps=normalized_iterator.validation_dates,
    epochs=200,
    callbacks=callbacks,
)
