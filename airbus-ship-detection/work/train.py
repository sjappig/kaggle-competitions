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
    parser.add_argument('--queue', type=int, default=10, help='max size of queue used for prefetching batches')

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


def generate_batches(samples, batch_size):
    x_batch = None
    y_batch = None
    batch_idx = 0
    ctr = 0
    while True:
        for x, y in common.generate_x_y(samples):
            if x_batch is None or y_batch is None:
                x_batch = numpy.array([x] * batch_size)
                y_batch = numpy.array([y] * batch_size)

            if batch_idx == batch_size:
                yield (x_batch, y_batch)
                batch_idx = 0

            x_batch[batch_idx] = x
            y_batch[batch_idx] = y
            batch_idx += 1
            ctr += 1

        yield (x_batch[0:batch_idx], y_batch[0:batch_idx])


def get_steps_per_epoch(epoch_size, batch_size):
    return int(numpy.ceil(epoch_size / batch_size))


if __name__ == '__main__':
    args = parse_args()

    model = create_model()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adadelta',
        metrics=['accuracy', 'mae'],
    )

    model.summary()

    segments_df = common.read_segments()

    print('{} entries available'.format(len(segments_df)), end=' ')
    print('of which {} has ships'.format(segments_df.EncodedPixels.count()))

    samples = common.sample(segments_df, n=args.samples)
    sample_count = len(samples)

    val_len = int(args.val * sample_count)
    train_len = sample_count - val_len

    train_data = samples.iloc[:train_len]
    val_data = samples.iloc[train_len:]

    batch_size = args.batch
    steps_per_epoch = get_steps_per_epoch(train_len, batch_size)
    epochs = args.epochs
    max_queue_size = args.queue

    print('sample count: {}'.format(sample_count))
    print('train length: {}'.format(train_len))
    print('validation length: {}'.format(val_len))
    print('batch size: {}'.format(batch_size))
    print('steps per epoch: {}'.format(steps_per_epoch))
    print('epochs: {}'.format(epochs))
    print('max queue size: {}'.format(max_queue_size))

    model.fit_generator(
        generate_batches(train_data, batch_size),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        max_queue_size=max_queue_size,
        validation_data=generate_batches(val_data, batch_size),
        validation_steps=get_steps_per_epoch(val_len, batch_size),
    )

    common.save_model(model)

