import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import LSTM, BatchNormalization
from keras.utils import Sequence
from ocpa.algo.predictive_monitoring.obj import Feature_Storage
from sklearn.model_selection import train_test_split

from definitions import ROOT_DIR
from src.encoding.sequential import generate_trace_prefixes


class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.n = len(x_set)

    def __len__(self):
        return self.n // self.batch_size

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y


class LSTM_Model(tf.keras.Sequential):
    def __init__(self, input_shape, n_layers: int = 2, n_neurons: int = 100, activation='tanh', dropout_rate=.1):
        super().__init__()
        self.n_layers = n_layers
        self.n_neurons = n_neurons

        # build lstm layers
        if n_layers == 1:
            self.add(LSTM(n_neurons, implementation=2, input_shape=input_shape, dropout=dropout_rate,
                          activation=activation))
            self.add(BatchNormalization())
        else:
            for i in range(n_layers - 1):
                self.add(LSTM(n_neurons, implementation=2, input_shape=input_shape, dropout=dropout_rate,
                              return_sequences=True, activation=activation))
                self.add(BatchNormalization())
            self.add(LSTM(n_neurons, implementation=2, input_shape=input_shape, dropout=dropout_rate,
                          activation=activation))
            self.add(BatchNormalization())
        # build output layer
        self.add(tf.keras.layers.Dense(units=1, kernel_initializer='glorot_uniform', name='remaining_time'))

    def my_name(self):
        return f'LSTM_{self.n_layers}_{self.n_neurons}'


def training_loop(model: LSTM_Model, train_loader, val_loader, num_epochs: int, dataset_name: str):
    print(model.summary())
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    checkpoint = os.path.join(ROOT_DIR, 'models', dataset_name, model.my_name() + '_weights_best.h5')

    # Remove old checkpoint
    if os.path.exists(checkpoint):
        os.remove(checkpoint)

    # Create callbacks
    # early_stopping = EarlyStopping(patience=42)
    model_checkpoint = ModelCheckpoint(checkpoint, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',
                                   min_delta=0.0001, cooldown=0, min_lr=0)

    callbacks = [model_checkpoint, lr_reducer]  # [early_stopping, model_checkpoint, lr_reducer]

    # train the model
    model.fit(train_loader, validation_data=val_loader, callbacks=callbacks, epochs=num_epochs,
              use_multiprocessing=False, verbose=2)

    # load best weights
    model.load_weights(checkpoint)

    return model


def train_model_with_lstm(feature_storage: Feature_Storage, target: tuple, dataset_name: str, n_layers: int = 2,
                          n_neurons: int = 100, num_epochs: int = 100) -> (LSTM_Model, list):
    # generate trace prefixes
    training_prefixes, training_targets = generate_trace_prefixes(feature_storage=feature_storage,
                                                                  target=target,
                                                                  min_trace_length=4,
                                                                  max_trace_length=4,
                                                                  index_list=feature_storage.training_indices)
    test_prefixes, test_targets = generate_trace_prefixes(feature_storage=feature_storage,
                                                          target=target,
                                                          min_trace_length=4,
                                                          max_trace_length=4,
                                                          index_list=feature_storage.test_indices)
    logging.info(f'Generated {len(training_prefixes)} training and {len(test_prefixes)} test sequences')

    # convert to numpy arrays
    logging.info('Converting to numpy arrays')
    training_dfs = list(map(lambda x: pd.DataFrame(data=x, columns=feature_storage.event_features[1:]), training_prefixes))
    test_dfs = list(map(lambda x: pd.DataFrame(data=x, columns=feature_storage.event_features[1:]), test_prefixes))
    y_training = np.asarray(training_targets, dtype='float32')
    y_test = np.asarray(test_targets, dtype='float32')

    # pad sequences
    logging.info('Padding sequences')
    X_training = tf.keras.preprocessing.sequence.pad_sequences(training_dfs, padding='post', dtype='float32')
    X_test = tf.keras.preprocessing.sequence.pad_sequences(test_dfs, padding='post', dtype='float32')

    # to dataset loaders
    logging.info('Creating dataset loaders')
    X_train, X_val, y_train, y_val = train_test_split(X_training, y_training, test_size=0.2, random_state=42)
    train_gen = DataGenerator(X_train, y_train, 128)
    val_gen = DataGenerator(X_val, y_val, 128)

    # build model
    model = LSTM_Model(input_shape=(X_train.shape[1], X_train.shape[2]), n_layers=n_layers, n_neurons=n_neurons)

    # compile model
    optimizer = tf.optimizers.Adam()
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error', 'mae', 'mape'])

    # train model
    model = training_loop(model=model, train_loader=train_gen, val_loader=val_gen, num_epochs=num_epochs,
                          dataset_name=dataset_name)

    # evaluate model
    scores = model.evaluate(X_test, y_test, verbose=0)

    return (), scores
