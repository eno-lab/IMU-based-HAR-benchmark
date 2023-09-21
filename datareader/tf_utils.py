import tensorflow as tf

def _gen(X, y, batch_size, shuffle_on_epoch_start):
    """ sample implementation to avoid None shape issue of tf.keras.utils.sequence """
    _X_output_spec = tf.TensorSpec(shape=X.shape[1:], dtype=tf.float32)
    _y_output_spec = tf.TensorSpec(shape=y.shape[1:], dtype=tf.float32)

    _ix = np.arange(len(X))
    assert len(X) == len(y), f'Length of X and y is different, {len(X)=} and {len(y)0}'

    def __gen():
        if shuffle_on_epoch_start:
            np.random.shuffle(_ix)

        for i in _ix:
            yield X[i], y[i]

    gen = tf.data.Dataset.from_generator(_gen,
              output_signature=(_X_output_spec, _y_output_spec))

    bgen = gen.batch(self.batch_size)
    bgen.prefetch(tf.data.AUTOTUNE)
    return bgen


def gen_train_dataset(data_reader, batch_size, shuffle_on_epoch_start):
    return _gen(data_reader.X_train, data_reader.y_train, batch_size, shuffle_on_epoch_start)


def gen_valid_dataset(data_reader, batch_size, shuffle_on_epoch_start):
    return _gen(data_reader.X_valid, data_reader.y_valid, batch_size, shuffle_on_epoch_start)


def gen_test_dataset(data_reader, batch_size, shuffle_on_epoch_start):
    return _gen(data_reader.X_test, data_reader.y_test, batch_size, shuffle_on_epoch_start)

