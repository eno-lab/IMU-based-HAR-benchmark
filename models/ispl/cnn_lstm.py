from tensorflow.keras import Input
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPool1D, Concatenate, Activation, Add, GlobalAveragePooling1D, \
    Dense, LSTM, TimeDistributed, Reshape, BatchNormalization, Bidirectional, Flatten, MaxPooling1D, Dropout, \
    SeparableConv1D
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import CategoricalCrossentropy, Reduction
import tensorflow as tf

# CNN-LSTM Model
def cnn_lstm(x_shape,
             n_classes,
             out_loss, 
             out_activ, 
             n_hidden=128,
             learning_rate=0.01,
             n_steps=4,
             length=32,
             n_signals=9,
             regularization_rate=0.01,
             cnn_depth=3,
             lstm_depth=2,
             metrics=['accuracy']):
    """ CNN1D_LSTM version 1: Divide 1 window into several smaller frames, then apply CNN to each frame
    - Input data format: [None, n_frames, n_timesteps, n_signals]"""

    _input_shape = x_shape[1:]
    m = Sequential()

    m.add(Reshape((n_steps, length, n_signals), input_shape=_input_shape))
    m.add(BatchNormalization())
    m.add(TimeDistributed(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')))
    m.add(TimeDistributed(Conv1D(filters=64, kernel_size=5, activation='relu')))
    m.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    m.add(TimeDistributed(Conv1D(filters=64, kernel_size=5, activation='relu')))
    m.add(TimeDistributed(Conv1D(filters=32, kernel_size=3, activation='relu')))
    m.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    m.add(TimeDistributed(Flatten()))
    for _ in range(lstm_depth-1):
        m.add(LSTM(n_hidden, return_sequences=True,
                   kernel_regularizer=l2(regularization_rate)))
    m.add(LSTM(n_hidden))
    m.add(Dropout(0.5))
    m.add(Dense(100, activation='relu',
                kernel_regularizer=l2(regularization_rate)))
    m.add(Dense(n_classes, activation=out_activ))

    m.compile(loss=out_loss,
              optimizer=Adam(learning_rate=learning_rate, amsgrad=True),
              weighted_metrics=metrics)
    return m


def gen_preconfiged_model(input_shape, n_classes, out_loss, out_activ, dataset, metrics=['accuracy']):

    # imported from tsf 
    n_steps = -1
    for n in [4, 3, 5, 2]:
        if input_shape[1] % n == 0:
            n_steps = n
            break

    if n_steps < 0:
        raise ValueError("The length of input cannot be divided by {n_steps}")

    length = int(input_shape[1]//n_steps)

    # Give model specific configurations
    hyperparameters = {'n_hidden': 512, 'n_steps': n_steps, 'length': length, 'n_signals': input_shape[-1],
                       'learning_rate': 0.0005, 'cnn_depth': 3, 'lstm_depth': 2,
                       'regularization_rate': 0.000093}

    return cnn_lstm(input_shape, n_classes, out_loss, out_activ, metrics=metrics, **hyperparameters), hyperparameters


def get_dnn_framework_name():
    return 'tensorflow'


