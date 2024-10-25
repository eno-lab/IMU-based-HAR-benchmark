import tensorflow as tf

IS_KERAS_VERSION_GE_3 = False
if hasattr(tf.keras, "version") and callable(tf.keras.version):
    IS_KERAS_VERSION_GE_3 = int(tf.keras.version().split('.')[0]) >= 3
