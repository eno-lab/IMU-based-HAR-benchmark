from tensorflow.keras import Input
from tensorflow.keras.layers import Conv1D, Concatenate, Dense, \
        TimeDistributed, BatchNormalization, Flatten, MaxPooling1D,\
        Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf

#
# Dua, N., Singh, S.N., Semwal, V.B. et al. 
# Inception inspired CNN-GRU hybrid network for human activity recognition. 
# Multimed Tools Appl 82, 5369–5403 (2023). https://doi.org/10.1007/s11042-021-11885-x
# 

from tensorflow.keras.layers import GRU
def icg_net(x_shape,
           n_classes,
           out_loss, 
           out_activ, 
           learning_rate=0.01,
           regularization_rate=0.01,
           metrics=['accuracy']):

    _input_shape = x_shape[1:]

    inputs = Input(_input_shape)
    x = inputs
    x = tf.expand_dims(x, 3)

    x1 = Sequential([
        TimeDistributed(Conv1D(32, 1, padding="same", activation="relu")),
        Dropout(0.3),
        TimeDistributed(Conv1D(8, 1, padding="same", activation="relu"))
    ])(x)
    x2 = Sequential([
        TimeDistributed(Conv1D(64, 3, padding="same", activation="relu")),
        Dropout(0.3),
        TimeDistributed(Conv1D(16, 1, padding="same", activation="relu")),
    ])(x)
    x3 = Sequential([
        TimeDistributed(Conv1D(64, 5, padding="same", activation="relu")),
        Dropout(0.3),
        TimeDistributed(Conv1D(16, 1, padding="same", activation="relu")),
    ])(x)
    x4 = Sequential([
        TimeDistributed(Conv1D(64, 11, padding="same", activation="relu")), 
        Dropout(0.3),
        TimeDistributed(Conv1D(16, 1, padding="same", activation="relu")),
    ])(x)
    x = Concatenate()([x1, x2, x3, x4])
    x = TimeDistributed(MaxPooling1D(2))(x)
    x = TimeDistributed(Flatten())(x)
    x = GRU(32, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = GRU(16, return_sequences=True)(x)
    x = Flatten()(x)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Dense(n_classes, activation=out_activ, kernel_regularizer=l2(regularization_rate))(x)

    m = Model(inputs, x)

    m.compile(loss=out_loss,
              optimizer=Adam(learning_rate=learning_rate, amsgrad=True),
              weighted_metrics=metrics)

    return m


def get_config(dataset, lr_magnif=1):
    # based on the original article
    return {'learning_rate': 0.001 * lr_magnif, 'regularization_rate': None}


def gen_model(input_shape, n_classes, out_loss, out_activ, metrics, config):
    return icg_net(input_shape, n_classes, out_loss, out_activ, metrics=metrics, **config)


def gen_preconfiged_model(input_shape, n_classes, out_loss, out_activ, dataset, metrics=['accuracy'], lr_magnif=1):
    config = get_config(dataset, lr_magnif)
    return gen_model(input_shape, n_classes, out_loss, out_activ, metrics, config), config


def get_optim_config(dataset, trial, lr_magnif=1):
    raise NotImplementedError("No config for optimization")


def get_dnn_framework_name():
    return 'tensorflow'
