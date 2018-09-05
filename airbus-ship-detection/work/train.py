import itertools
import os.path

from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Dense, Flatten, MaxPooling2D, Reshape
from keras.models import Model, Sequential
import numpy

from . import common

def get_feature_extraction_model(input_shape):
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


if __name__ == '__main__':
    segments_df = common.read_segments()

    print('{} entries available'.format(len(segments_df)), end=' ')
    print('of which {} has ships'.format(segments_df.EncodedPixels.count()))

    samples = common.sample(segments_df, n=2048)

    x, y = zip(*common.generate_x_y(samples))

    x_train = numpy.array(x)
    y_train = numpy.array(y)

    print('Input shape: {}'.format(x_train.shape))
    print('Output shape: {}'.format(y_train.shape))

    all_pixels = len(y_train) * 768 * 768
    ship_pixels = numpy.sum(y_train)
    blank_pixels = all_pixels - ship_pixels

    print('{} pixels, {} has ship, {} do not'.format(all_pixels, ship_pixels, blank_pixels))

    feat_model = get_feature_extraction_model((768, 768, 3))
    output = Conv2DTranspose(filters=3, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(feat_model.output)
    output = BatchNormalization()(output)
    output = Conv2DTranspose(filters=3, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(output)
    output = BatchNormalization()(output)
    output = Conv2DTranspose(filters=3, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(output)
    output = BatchNormalization()(output)
    output = Conv2DTranspose(filters=1, kernel_size=3, strides=(2, 2), padding='same', activation='sigmoid')(output)
    output = Reshape((768, 768))(output)

    model = Model(
        inputs=[feat_model.input],
        outputs=[output],
    )

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adadelta',
        metrics=['accuracy', 'mae'],
    )

    model.summary()

    model.fit(
        x_train,
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
    )

    common.save_model(model)

