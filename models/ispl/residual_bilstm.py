from tensorflow.keras import Input
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPool1D, Concatenate, Activation, Add, GlobalAveragePooling1D, \
    Dense, LSTM, TimeDistributed, Reshape, BatchNormalization, Bidirectional, Flatten, MaxPooling1D, Dropout, \
    SeparableConv1D
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import CategoricalCrossentropy, Reduction
import tensorflow as tf


# Residual BiLSTM Model
# Yong Li and andLuping Wang, Human Activity Recognition Based on Residual Network and BiLSTM
def residual_bilstm(x_shape,
           n_classes,
           out_loss, 
           out_activ, 
           n_hidden=64,
           learning_rate=0.00003,
           #regularization_rate=0.01,
           metrics=['accuracy']):
    """ Requires 3D data: [n_samples, n_timesteps, n_features]"""

    _input_shape = x_shape[1:]
    inputs = Input(_input_shape)
    inputs = tf.expand_dims(inputs, 3) 

    x = Conv2D(32, (2,2), strides=2, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = Conv2D(32, (2,2), strides=1, padding="same")(x)
    x = BatchNormalization()(x)

    x2 = Conv2D(32, (1, 1), strides=2, padding="same")(inputs)
    x2 = BatchNormalization()(x2)
    x = Add()([x, x2])
    x = tf.keras.activations.relu(x)
    x = Dropout(0.5)(x)
    x = TimeDistributed(Flatten())(x)
    x = Bidirectional(LSTM(n_hidden), merge_mode='concat')(x)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, activation=out_activ)(x)

    m = Model(inputs=inputs, outputs=x)

    m.compile(loss=out_loss,
              optimizer=Adam(learning_rate=learning_rate, amsgrad=True),
              weighted_metrics=metrics)

    return m


def get_config(dataset, lr_magnif=1):
    return {'n_hidden': 64, 'learning_rate': (0.00003 if dataset.startswith("pamap2") else 0.0001)*lr_magnif}


def gen_model(input_shape, n_classes, out_loss, out_activ, metrics, config):
    return residual_bilstm(input_shape, n_classes, out_loss, out_activ, metrics=metrics, **config)


def gen_preconfiged_model(input_shape, n_classes, out_loss, out_activ, dataset, metrics=['accuracy'], lr_magnif=1):
    config = get_config(dataset, lr_magnif)
    return gen_model(input_shape, n_classes, out_loss, out_activ, metrics, config), config


def get_optim_config(dataset, trial, lr_magnif=1):
    raise NotImplementedError("No config for optimization")


def get_dnn_framework_name():
    return 'tensorflow'
