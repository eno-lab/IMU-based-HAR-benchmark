from tensorflow.keras import Input
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPool1D, Concatenate, Activation, Add, GlobalAveragePooling1D, \
    Dense, LSTM, TimeDistributed, Reshape, BatchNormalization, Bidirectional, Flatten, MaxPooling1D, Dropout, \
    SeparableConv1D
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import CategoricalCrossentropy, Reduction
import tensorflow as tf


# BiLSTM Model
def bilstm(x_shape,
           n_classes,
           out_loss, 
           out_activ, 
           n_hidden=128,
           learning_rate=0.01,
           regularization_rate=0.01,
           merge_mode='concat',
           depth=2,
           metrics=['accuracy']):
    """ Requires 3D data: [n_samples, n_timesteps, n_features]"""

    _input_shape = x_shape[1:]
    m = Sequential()

    m.add(BatchNormalization(input_shape=_input_shape))
    m.add(Bidirectional(LSTM(n_hidden), merge_mode=merge_mode))
    m.add(Dense(100, activation='relu', kernel_regularizer=l2(regularization_rate)))
    m.add(Dense(n_classes, activation=out_activ))

    m.compile(loss=out_loss,
              optimizer=Adam(learning_rate=learning_rate, amsgrad=True),
              weighted_metrics=metrics)

    return m


def gen_preconfiged_model(input_shape, n_classes, out_loss, out_activ, dataset, metrics=['accuracy']):
    # Give model specific configurations
    hyperparameters = {'n_hidden': 128, 'learning_rate': 0.0005, 'regularization_rate': 0.000093,
                       'depth': 1,
                       'merge_mode': 'concat'}
    return bilstm(input_shape, n_classes, out_loss, out_activ, metrics=metrics, **hyperparameters), hyperparameters