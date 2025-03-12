import warnings
import numpy as np

IS_KERAS_VERSION_GE_3 = False
__import_success = False
try:
    import keras
    __import_success = True
except ModuleNotFoundError as e:
    try:
        from tensorflow import keras
        __import_success = True
    except ModuleNotFoundError as e:
        pass

if __import_success:
    if hasattr(keras, "version") and callable(keras.version):
        IS_KERAS_VERSION_GE_3 = int(keras.version().split('.')[0]) >= 3

class StopWithNan(keras.callbacks.Callback):

    def __init__(self, monitor="val_loss", **kwargs):
        super().__init__(**kwargs)

        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        target = logs.get(self.monitor)

        if target is None:
            warnings.warn(
                f"`{self.monitor}` which is not available. Available metrics "
                f"are: {','.join(list(logs.keys()))}.",
                stacklevel=2,
            )
            return
        
        if np.isnan(target):
            self.model.stop_training = True
