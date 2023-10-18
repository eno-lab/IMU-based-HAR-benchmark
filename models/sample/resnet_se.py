from tensorflow.keras import Input
from tensorflow.keras.layers import Conv1D, Activation, Add, GlobalAveragePooling1D, \
    Dense, BatchNormalization, Flatten, MaxPooling1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf

#
# S. Mekruksavanich, A. Jitpattanakul, K. Sitthithakerngkiet, P. Youplao and P. Yupapin, 
# "ResNet-SE: Channel Attention-Based Deep Residual Network for Complex Activity Recognition 
# Using Wrist-Worn Wearable Sensors," in IEEE Access, vol. 10, pp. 51142-51154, 2022, 
# doi: 10.1109/ACCESS.2022.3174124.
#
 
def resnet_se(x_shape,
              n_classes,
              out_loss, 
              out_activ, 
              learning_rate=0.01,
              regularization_rate=0.01,
              metrics=['accuracy']):

    _input_shape = x_shape[1:]

    inputs = Input(_input_shape)
    x = inputs

    x = Sequential([
        Conv1D(64, 5, padding="same"),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(2)
        ])(x)

    for i in range(5):
        x_in = x
        x = Sequential([
            Conv1D(32, 5, padding="same"),
            BatchNormalization(),
            Activation('relu'),
            Conv1D(64, 5, padding="same"),
            BatchNormalization()
            ])(x)

        se_pram = Sequential([
            GlobalAveragePooling1D(),
            Dense(128, activation="relu"),
            Dense(64, activation="sigmoid"),
            ])(x)

        se_pram = tf.expand_dims(se_pram, 1)
        x = tf.multiply(x, se_pram)
        x = Add()([x, x_in])
        x = Activation('relu')(x)

    x = Sequential([
        GlobalAveragePooling1D(),
        Flatten(),
        Dense(128, activation="relu"),
        ])(x)

    x = Dense(n_classes, activation='softmax', kernel_regularizer=l2(regularization_rate))(x)

    m = Model(inputs, x)

    m.compile(loss=out_loss,
              optimizer=Adam(learning_rate=learning_rate, amsgrad=True),
              weighted_metrics=metrics)

    return m


def get_config(dataset, lr_magnif=1):
    # based on the original article
    return {'learning_rate': 0.001 * lr_magnif, 'regularization_rate': None}


def gen_model(input_shape, n_classes, out_loss, out_activ, metrics, config):
    return resnet_se(input_shape, n_classes, out_loss, out_activ, metrics=metrics, **config)


def gen_preconfiged_model(input_shape, n_classes, out_loss, out_activ, dataset, metrics=['accuracy'], lr_magnif=1):
    config = get_config(dataset, lr_magnif)
    return gen_model(input_shape, n_classes, out_loss, out_activ, metrics, config), config


def get_optim_config(dataset, trial, lr_magnif=1):
    raise NotImplementedError("No config for optimization")


def get_dnn_framework_name():
    return 'tensorflow'
