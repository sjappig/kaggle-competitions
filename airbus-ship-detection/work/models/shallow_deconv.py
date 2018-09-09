from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Dense, Flatten, MaxPooling2D, Reshape
from keras.models import Model, Sequential


def create(input_shape):
    feat_model = _get_feature_extraction_model(input_shape)
    output = Conv2DTranspose(filters=3, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(feat_model.output)
    output = BatchNormalization()(output)
    output = Conv2DTranspose(filters=3, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(output)
    output = BatchNormalization()(output)
    output = Conv2DTranspose(filters=3, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(output)
    output = BatchNormalization()(output)
    output = Conv2DTranspose(filters=1, kernel_size=3, strides=(2, 2), padding='same', activation='sigmoid')(output)
    output = Reshape(input_shape[:2])(output)

    return Model(
        inputs=[feat_model.input],
        outputs=[output],
    )


def _get_feature_extraction_model(input_shape):
    model = Sequential()
    model.add(Conv2D(filters=3, kernel_size=3, padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(filters=3, kernel_size=3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(filters=3, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=3, kernel_size=3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(filters=3, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=3, kernel_size=3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(filters=9, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=9, kernel_size=3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))

    return model
