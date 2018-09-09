from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, Reshape
from keras.models import Model, Sequential


def create(input_shape):
    feat_model = _get_feature_extraction_model(input_shape)
    transpose_args = {
        'kernel_size': 3,
        'strides': (2, 2),
        'padding': 'same',
    }

    output = Conv2DTranspose(filters=3, activation='relu', **transpose_args)(feat_model.output)
    output = BatchNormalization()(output)
    output = Conv2DTranspose(filters=3, activation='relu', **transpose_args)(output)
    output = BatchNormalization()(output)
    output = Conv2DTranspose(filters=3, activation='relu', **transpose_args)(output)
    output = BatchNormalization()(output)
    output = Conv2DTranspose(filters=1, activation='sigmoid', **transpose_args)(output)
    output = Reshape(input_shape[:2])(output)

    return Model(
        inputs=[feat_model.input],
        outputs=[output],
    )


def _get_feature_extraction_model(input_shape):
    model = Sequential()
    conv_args = {
        'kernel_size': 3,
        'padding': 'same',
        'activation': 'relu',
    }

    model.add(Conv2D(filters=3, input_shape=input_shape, **conv_args))
    model.add(Conv2D(filters=3, **conv_args))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(filters=3, **conv_args))
    model.add(Conv2D(filters=3, **conv_args))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(filters=3, **conv_args))
    model.add(Conv2D(filters=3, **conv_args))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(filters=9, **conv_args))
    model.add(Conv2D(filters=9, **conv_args))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))

    return model
