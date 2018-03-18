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

validate_size = 100000
DATA_FILE_LENGTH=3453424
FeatureProcessor = feature_nn_transform.FeatureTfIdfTransformer(
    'lemmas/coursed_lemmas.txt', 'lemmas/wc.txt', 'course.csv',
    validate_size, standartize_path='nn_feature_processor.npz',
    total_document_count=DATA_FILE_LENGTH-validate_size)

model = keras.models.Sequential()
model.add(keras.layers.Dense(512, input_shape=(FeatureProcessor.vector_size,), activation='elu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(128, activation='elu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])

batch_size = 128
epochs = 5000

callbacks = [
    keras.callbacks.ModelCheckpoint(filepath='nn/2_dense_03_drop_model.hdf5', save_best_only=True, period=1),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=20, verbose=1, cooldown=10),
    keras.callbacks.CSVLogger('nn/2_dense_03_drop_model.csv'),
    keras.callbacks.EarlyStopping(patience=50)
]

input('Press ENTER to start')

model.fit_generator(
    FeatureProcessor.iterate_train_data(batch_size),
    FeatureProcessor.total_document_count // batch_size,
    validation_data=FeatureProcessor.iterate_validation_data(batch_size),
    validation_steps=validate_size // batch_size,
    epochs=epochs,
    callbacks=callbacks,
)
