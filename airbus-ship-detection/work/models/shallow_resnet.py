import keras.layers
from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Input, MaxPooling2D, Reshape
from keras.models import Model


def create(input_shape):
    feat_model = _get_feature_extraction_model(input_shape)
    transpose_args = {
        'kernel_size': 3,
        'strides': (2, 2),
        'padding': 'same',
    }

    output = Conv2DTranspose(filters=3, activation='relu', **transpose_args)(feat_model['output'])
    output = BatchNormalization()(output)
    output = Conv2DTranspose(filters=3, activation='relu', **transpose_args)(output)
    output = BatchNormalization()(output)
    output = Conv2DTranspose(filters=3, activation='relu', **transpose_args)(output)
    output = BatchNormalization()(output)
    output = Conv2DTranspose(filters=1, activation='sigmoid', **transpose_args)(output)
    output = Reshape(input_shape[:2])(output)

    return Model(
        inputs=[feat_model['input']],
        outputs=[output],
    )


def _get_feature_extraction_model(input_shape):
    model = {}
    conv_args = {
        'kernel_size': 3,
        'padding': 'same',
        'activation': 'relu',
    }

    model['input'] = Input(shape=input_shape)
    input_layer = model['input']
    layer = Conv2D(filters=3, input_shape=input_shape, **conv_args)(input_layer)
    layer = Conv2D(filters=3, **conv_args)(layer)
    layer = keras.layers.add([input_layer, layer])
    layer = BatchNormalization()(layer)
    layer = MaxPooling2D(2, 2)(layer)

    input_layer = layer
    layer = Conv2D(filters=3, input_shape=input_shape, **conv_args)(layer)
    layer = Conv2D(filters=3, **conv_args)(layer)
    layer = keras.layers.add([input_layer, layer])
    layer = BatchNormalization()(layer)
    layer = MaxPooling2D(2, 2)(layer)

    input_layer = layer
    layer = Conv2D(filters=3, input_shape=input_shape, **conv_args)(layer)
    layer = Conv2D(filters=3, **conv_args)(layer)
    layer = keras.layers.add([input_layer, layer])
    layer = BatchNormalization()(layer)
    layer = MaxPooling2D(2, 2)(layer)

    layer = Conv2D(filters=9, input_shape=input_shape, **conv_args)(layer)
    layer = Conv2D(filters=9, **conv_args)(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPooling2D(2, 2)(layer)

    model['output'] = layer

    return model
