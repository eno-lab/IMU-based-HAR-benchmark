from tensorflow.keras import Input
from tensorflow.keras.layers import Conv1D, Concatenate, Dense, \
        BatchNormalization, MaxPooling1D, GRU, GlobalAveragePooling1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf

#
# L. Lu, C. Zhang, K. Cao, T. Deng and Q. Yang, 
# "A Multichannel CNN-GRU Model for Human Activity Recognition," 
# in IEEE Access, vol. 10, pp. 66797-66810, 2022, doi: 10.1109/ACCESS.2022.3185112.
# 

def mch_cnn_gru(x_shape,
           n_classes,
           out_loss, 
           out_activ, 
           learning_rate=0.001,
           metrics=['accuracy']):

    _input_shape = x_shape[1:]

    inputs = Input(_input_shape)


    def gen_ch(stride):
        s = Sequential()
        s.add(Conv1D(64, stride, activation='relu'))
        s.add(BatchNormalization())
        s.add(Conv1D(128, stride, activation='relu'))
        s.add(MaxPooling1D(2))
        return s

    l = []
    for st in [3, 5, 7]:
        l.append(gen_ch(st)(inputs))

    x = Concatenate(axis=1)(l)
    x = GRU(128, return_sequences=True)(x)
    x = GRU(64, return_sequences=True)(x)
    x = GlobalAveragePooling1D()(x)
    x = BatchNormalization()(x)
    x = Dense(n_classes, activation='softmax', 
            kernel_regularizer=l2(regularization_rate)
            )(x)

    m = Model(inputs, x)

    m.compile(loss=out_loss,
              optimizer=Adam(learning_rate=learning_rate, amsgrad=True),
              weighted_metrics=metrics)

    return m


def get_config(dataset, lr_magnif=1):
    # based on the original article
    return {'learning_rate': 0.001 * lr_magnif, 'regularization_rate': None}


def gen_model(input_shape, n_classes, out_loss, out_activ, metrics, config):
    return mch_cnn_gru(input_shape, n_classes, out_loss, out_activ, metrics=metrics, **config)


def gen_preconfiged_model(input_shape, n_classes, out_loss, out_activ, dataset, metrics=['accuracy'], lr_magnif=1):
    config = get_config(dataset, lr_magnif)
    return gen_model(input_shape, n_classes, out_loss, out_activ, metrics, config), config


def get_optim_config(dataset, trial, lr_magnif=1):
    raise NotImplementedError("No config for optimization")


def get_dnn_framework_name():
    return 'tensorflow'
