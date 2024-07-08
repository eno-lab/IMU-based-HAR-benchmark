from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D, Dropout, Dense, LSTM
from tensorflow.keras.regularizers import l2

def get_config(dataset, lr_magnif=1):
    return {'n_hidden': 128,
            'n_filters': 64,
            'filter_size': 5,
            'drop_prob': 0.5,
            'cnn_depth': 4,
            'lstm_depth': 2,
            'learning_rate': 0.001 * lr_magnif,
            'regularization_rate': 0.01}


def deep_conv_lstm(input_shape, 
              n_classes, 
              out_loss, 
              out_activ, 
              n_hidden=128, 
              n_filters=64, 
              filter_size=5, 
              drop_prob=0.5, 
              learning_rate=0.01, 
              cnn_depth=4,
              lstm_depth=2,
              metrics=['accuracy'],
              regularization_rate=0.01
              ):
    _input_shape = input_shape[1:]

    inputs = Input(_input_shape)
    x = inputs

    for _ in range(cnn_depth):
        x = Conv1D(n_filters, filter_size, activation='relu')(x)
    for _ in range(lstm_depth):
        x = LSTM(n_hidden, return_sequences=True)(x)
    
    x = Dropout(drop_prob)(x)
    x = x[:, -1, :] # extract values of final step 
    x = Dense(n_classes, activation=out_activ, 
            kernel_regularizer=l2(regularization_rate))(x) 

    m = Model(inputs, x)
    m.compile(loss=out_loss,
            optimizer=Adam(learning_rate=learning_rate, amsgrad=True),
            weighted_metrics=metrics)

    return m


def gen_model(input_shape, n_classes, out_loss, out_activ, metrics, config):
    return deep_conv_lstm(input_shape, n_classes, out_loss, out_activ, metrics=metrics, **config)


def gen_preconfiged_model(input_shape, n_classes, out_loss, out_activ, dataset, metrics=['accuracy'], lr_magnif=1):
    config = get_config(dataset, lr_magnif)
    return gen_model(input_shape, n_classes, out_loss, out_activ, metrics, config), config


def get_optim_config(dataset, trial, lr_magnif=1):
    raise NotImplementedError("No config for optimization")


def get_dnn_framework_name():
    return 'tensorflow'
