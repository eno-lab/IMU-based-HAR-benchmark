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
