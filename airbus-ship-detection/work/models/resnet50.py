from keras.applications.resnet50 import ResNet50
from keras.layers import BatchNormalization, Conv2DTranspose, Reshape
from keras.models import Model


def create(input_shape):
    feat_model = _get_feature_extraction_model(input_shape)
    transpose_args = {
        'kernel_size': 3,
        'strides': (2, 2),
        'padding': 'same',
    }

    output = Conv2DTranspose(filters=256, activation='relu', **transpose_args)(feat_model.output)
    output = BatchNormalization()(output)
    output = Conv2DTranspose(filters=64, activation='relu', **transpose_args)(output)
    output = BatchNormalization()(output)
    output = Conv2DTranspose(filters=32, activation='relu', **transpose_args)(output)
    output = BatchNormalization()(output)
    output = Conv2DTranspose(filters=8, activation='relu', **transpose_args)(output)
    output = BatchNormalization()(output)
    output = Conv2DTranspose(filters=1, activation='sigmoid', **transpose_args)(output)
    output = Reshape(input_shape[:2])(output)

    return Model(
        inputs=[feat_model.input],
        outputs=[output],
    )


def _get_feature_extraction_model(input_shape):
    model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    return model
