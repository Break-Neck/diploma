#!/usr/bin/env python3

import numpy as np
import keras
import sys

import logging

if len(sys.argv) >= 2:
    logger = logging.getLogger()
    if sys.argv[-1].lower() == 'info':
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)


import feature_nn_transform

train_part = 0.8
validate_part = 0.1
data_processor = feature_nn_transform.CountTruncatedVectorizer('lemmas/ccl.txt', 'lemmas/wc.txt', 'course.csv', train_part, validate_part, dates_lines_starts_load=True, scaler_start_load=True)

model = keras.models.Sequential()
model.add(keras.layers.LSTM(256, input_shape=(None, data_processor.vector_length)))#, return_sequences=True))
#model.add(keras.layers.Dropout(0.2))
#model.add(keras.layers.LSTM(64))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['acc'])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint(filepath='nn/model.hdf5', save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, verbose=1, cooldown=10),
    keras.callbacks.CSVLogger('nn/model.csv'),
    keras.callbacks.EarlyStopping(patience=25)
]

input('Press ENTER to start')

model.fit_generator(
    data_processor.iterate_train_chunk(),
    data_processor.train_dates,
    validation_data=data_processor.iterate_validation_chunk(),
    validation_steps=data_processor.validation_dates,
    epochs=200,
    callbacks=callbacks,
)
