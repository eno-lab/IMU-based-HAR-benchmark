from .hart import mobileHART_XS
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

def get_config(dataset, lr_magnif=1):
    return {'projectionDims': [96,120,144],
            'filterCount': [16//2,32//2,48//2,64//2,80,96,384],
            'expansion_factor': 4,
            'mlp_head_units': [1024],
            'dropout_rate': 0.3,
            'learning_rate': 0.001 * lr_magnif,
            'regularization_rate': 0.00001}


def gen_model(input_shape, n_classes, out_loss, out_activ, metrics, config):

    m = mobileHART_XS(input_shape, activityCount=n_classes, **config)

    m.compile(loss=out_loss,
            optimizer=Adam(learning_rate=config['learning_rate'], amsgrad=True),
            weighted_metrics=metrics)

    return m


def gen_preconfiged_model(input_shape, n_classes, out_loss, out_activ, dataset, metrics=['accuracy'], lr_magnif=1):
    config = get_config(dataset, lr_magnif)
    return gen_model(input_shape, n_classes, out_loss, out_activ, metrics, config), config


def get_optim_config(dataset, trial, lr_magnif=1):
    raise NotImplementedError("No config for optimization")


def get_dnn_framework_name():
    return 'tensorflow'
