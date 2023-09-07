# title       :models
# description :Script that contains the models used in our experiments.
# author      :Ronald Mutegeki
# date        :20210203
# version     :1.0
# usage       :Call it in utils.py.
# notes       :This script is where the dataset we are using in this experiment is defined.
#              You can customize the loss and activation functions, and any other model related configs.
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPool1D, Concatenate, Activation, Add, GlobalAveragePooling1D, \
    Dense, LSTM, TimeDistributed, Reshape, BatchNormalization, Bidirectional, Flatten, MaxPooling1D, Dropout, \
    SeparableConv1D
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import CategoricalCrossentropy, Reduction
import tensorflow as tf


out_loss = 'binary_crossentropy'
out_activ = 'sigmoid'

def init_loss_and_activation(dataset):
    global out_loss, out_activ
    if dataset == 'daphnet':
        out_loss = 'binary_crossentropy'
        out_activ = 'sigmoid'
    else:
        out_loss = CategoricalCrossentropy(reduction=Reduction.AUTO, name='output_loss')
        out_activ = 'softmax'

# CNN Model
def cnn(x_shape,
        n_classes,
        filters,
        fc_hidden_nodes,
        learning_rate=0.01, regularization_rate=0.01,
        metrics=None):
    if metrics is None:
        metrics = ['accuracy']
    dim_length = x_shape[1]  # number of samples in a time series
    dim_channels = x_shape[2]  # number of channels
    dim_output = n_classes
    weightinit = 'lecun_uniform'  # weight initialization
    m = Sequential()

    m.add(BatchNormalization(input_shape=(dim_length, dim_channels)))
    for filter_number in filters:
        m.add(Conv1D(filter_number, kernel_size=3, padding='same',
                     kernel_regularizer=l2(regularization_rate),
                     kernel_initializer=weightinit))
        m.add(BatchNormalization())
        m.add(Activation('relu'))
    m.add(Flatten())
    m.add(Dense(units=fc_hidden_nodes,
                kernel_regularizer=l2(regularization_rate),
                kernel_initializer=weightinit))  # Fully connected layer
    m.add(Activation('relu'))  # Relu activation
    m.add(Dense(units=dim_output, kernel_initializer=weightinit, kernel_regularizer=l2(regularization_rate)))
    m.add(BatchNormalization())
    m.add(Activation(out_activ))  # Final classification layer

    m.compile(loss=out_loss,
              optimizer=Adam(lr=learning_rate),
              weighted_metrics=metrics)

    return m


# CNN-LSTM Model
def cnn_lstm(x_shape,
             n_classes,
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


# Vanilla LSTM Model
def vanilla_lstm(x_shape,
                 n_classes,
                 n_hidden=128,
                 learning_rate=0.01,
                 regularization_rate=0.01,
                 metrics=['accuracy']):
    """ Requires 3D data: [n_samples, n_timesteps, n_signals]"""
    _input_shape = x_shape[1:]
    m = Sequential()

    m.add(BatchNormalization(input_shape=_input_shape))
    m.add(LSTM(n_hidden))
    m.add(Dropout(0.3))
    m.add(Dense(100, activation='relu'))
    m.add(Dense(n_classes, activation=out_activ, kernel_regularizer=l2(regularization_rate)))

    m.compile(loss=out_loss,
              optimizer=Adam(learning_rate=learning_rate),
              weighted_metrics=metrics)
    return m


# Stacked LSTM Model
def stacked_lstm(x_shape,
                 n_classes,
                 n_hidden=128,
                 learning_rate=0.01,
                 regularization_rate=0.01,
                 depth=2,
                 metrics=['accuracy']):
    """ Require 3D data: [n_samples, n_timesteps, n_signals]"""
    _input_shape = x_shape[1:]
    dim_length = x_shape[1]  # number of samples in a time series
    dim_channels = x_shape[2]  # number of channels
    dim_output = n_classes
    m = Sequential()

    m.add(BatchNormalization(input_shape=_input_shape))
    m.add(Dense(100, activation='relu', name='preprocess', kernel_regularizer=l2(regularization_rate)))
    m.add(LSTM(n_hidden, return_sequences=True, kernel_regularizer=l2(regularization_rate)))
    m.add(Dropout(0.5))
    m.add(LSTM(n_hidden))
    m.add(Dense(100, activation='relu'))
    m.add(Dense(dim_output, activation=out_activ, name="output"))

    m.compile(loss=out_loss,
              optimizer=Adam(learning_rate=learning_rate, amsgrad=True),
              weighted_metrics=metrics)
    return m


# BiLSTM Model
def bilstm(x_shape,
           n_classes,
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

# Residual BiLSTM Model
# Yong Li and andLuping Wang, Human Activity Recognition Based on Residual Network and BiLSTM
def residual_bilstm(x_shape,
           n_classes,
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

def bilstm(x_shape,
           n_classes,
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


# iSPLInception Model
def ispl_inception(x_shape,
                   n_classes,
                   filters_number,
                   network_depth=5,
                   use_residual=True,
                   use_bottleneck=True,
                   max_kernel_size=20,
                   learning_rate=0.01,
                   bottleneck_size=32,
                   regularization_rate=0.01,
                   metrics=['accuracy']):
    dim_length = x_shape[1]  # number of samples in a time series
    dim_channels = x_shape[2]  # number of channels
    weightinit = 'lecun_uniform'  # weight initialization

    def inception_module(input_tensor, stride=1, activation='relu'):

        # The  channel number is greater than 1
        if use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = Conv1D(filters=bottleneck_size,
                                     kernel_size=1,
                                     padding='same',
                                     activation=activation,
                                     kernel_initializer=weightinit,

                                     use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        kernel_sizes = [max_kernel_size // (2 ** i) for i in range(3)]
        conv_list = []

        for kernel_size in kernel_sizes:
            conv_list.append(Conv1D(filters=filters_number,
                                    kernel_size=kernel_size,
                                    strides=stride,
                                    padding='same',
                                    activation=activation,
                                    kernel_initializer=weightinit,
                                    kernel_regularizer=l2(regularization_rate),
                                    use_bias=False)(input_inception))

        max_pool_1 = MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_last = Conv1D(filters=filters_number,
                           kernel_size=1,
                           padding='same',
                           activation=activation,
                           kernel_initializer=weightinit,
                           kernel_regularizer=l2(regularization_rate),
                           use_bias=False)(max_pool_1)

        conv_list.append(conv_last)

        x = Concatenate(axis=2)(conv_list)
        x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)
        return x

    def shortcut_layer(input_tensor, out_tensor):
        shortcut_y = Conv1D(filters=int(out_tensor.shape[-1]),
                            kernel_size=1,
                            padding='same',
                            kernel_initializer=weightinit,
                            kernel_regularizer=l2(regularization_rate),
                            use_bias=False)(input_tensor)
        shortcut_y = BatchNormalization()(shortcut_y)

        x = Add()([shortcut_y, out_tensor])
        x = Activation('relu')(x)
        return x

    # Build the actual model:
    input_layer = Input((dim_length, dim_channels))
    x = BatchNormalization()(input_layer)  # Added batchnorm (not in original paper)
    input_res = x

    for depth in range(network_depth):
        x = inception_module(x)

        if use_residual and depth % 3 == 2:
            x = shortcut_layer(input_res, x)
            input_res = x

    gap_layer = GlobalAveragePooling1D()(x)

    # Final classification layer
    output_layer = Dense(n_classes, activation=out_activ,
                         kernel_initializer=weightinit, kernel_regularizer=l2(regularization_rate))(gap_layer)

    # Create model and compile
    m = Model(inputs=input_layer, outputs=output_layer)

    m.compile(loss=out_loss,
              optimizer=Adam(learning_rate=learning_rate, amsgrad=True),
              weighted_metrics=metrics)

    return m


from tensorflow.keras.layers import Activation
from tensorflow.keras import activations
def ResNet_SE(x_shape,
              n_classes,
              learning_rate=0.01,
              regularization_rate=0.01,
              metrics=['accuracy']):

    _input_shape = x_shape[1:]

    inputs = Input(_input_shape)
    x = inputs

    x = Sequential([
        Conv1D(64, 5, padding="same"),
        BatchNormalization(),
        Activation(activations.relu),
        MaxPooling1D(2)
        ])(x)

    for i in range(5):
        x_in = x
        x = Sequential([
            Conv1D(32, 5, padding="same"),
            BatchNormalization(),
            Activation(activations.relu),
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
        x = Activation(activations.relu)(x)

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


from tensorflow.keras.layers import GRU
def icg_net(x_shape,
           n_classes,
           n_steps=4,
           length=32,
           n_signals=9,
           learning_rate=0.01,
           regularization_rate=0.01,
           metrics=['accuracy']):

    _input_shape = x_shape[1:]

    inputs = Input(_input_shape)
    x = inputs
    #x = Reshape((n_steps, length, n_signals))(x)
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
    print(x.shape)
    x = TimeDistributed(MaxPooling1D(2))(x)
    x = TimeDistributed(Flatten())(x)
    #x = tf.expand_dims(x, 2)
    x = GRU(32, return_sequences=True, kernel_regularizer=l2(regularization_rate))(x)
    x = Dropout(0.3)(x)
    #x = tf.expand_dims(x, 2)
    x = GRU(16)(x)
    x = Dense(64, kernel_regularizer=l2(regularization_rate))(x)
    x = BatchNormalization()(x)
    x = Dense(n_classes, activation='softmax', kernel_regularizer=l2(regularization_rate))(x)

    m = Model(inputs, x)

    m.compile(loss=out_loss,
              optimizer=Adam(learning_rate=learning_rate, amsgrad=True),
              weighted_metrics=metrics)

    return m
