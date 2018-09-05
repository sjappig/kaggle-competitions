import argparse
import itertools
import os.path

from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Dense, Flatten, MaxPooling2D, Reshape
from keras.models import Model, Sequential
import numpy

from . import common


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--samples', type=int, default=None, help='number of samples; if not given, use all')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--val', type=float, default=0.2, help='fraction of samples to use for validation')

    return parser.parse_args()


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


def create_model():
    feat_model = get_feature_extraction_model((768, 768, 3))
    output = Conv2DTranspose(filters=3, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(feat_model.output)
    output = BatchNormalization()(output)
    output = Conv2DTranspose(filters=3, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(output)
    output = BatchNormalization()(output)
    output = Conv2DTranspose(filters=3, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(output)
    output = BatchNormalization()(output)
    output = Conv2DTranspose(filters=1, kernel_size=3, strides=(2, 2), padding='same', activation='sigmoid')(output)
    output = Reshape((768, 768))(output)

    return Model(
        inputs=[feat_model.input],
        outputs=[output],
    )


if __name__ == '__main__':
    args = parse_args()

    segments_df = common.read_segments()

    print('{} entries available'.format(len(segments_df)), end=' ')
    print('of which {} has ships'.format(segments_df.EncodedPixels.count()))

    samples = common.sample(segments_df, n=args.samples)

    x, y = zip(*common.generate_x_y(samples))

    x_train = numpy.array(x)
    y_train = numpy.array(y)

    print('Input shape: {}'.format(x_train.shape))
    print('Output shape: {}'.format(y_train.shape))

    all_pixels = len(y_train) * 768 * 768
    ship_pixels = numpy.sum(y_train)
    blank_pixels = all_pixels - ship_pixels

    print('{} pixels, {} has ship, {} do not'.format(all_pixels, ship_pixels, blank_pixels))

    model = create_model()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adadelta',
        metrics=['accuracy', 'mae'],
    )

    model.summary()

    model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch,
        validation_split=args.val,
    )

    common.save_model(model)

