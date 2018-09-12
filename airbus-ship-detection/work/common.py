import itertools
import os.path

import keras.models
import keras.preprocessing
import numpy
import pandas


MODEL_FILENAME = 'shipmodel.h5'
SUBMISSION_FILENAME = 'submission.csv'

_TRAIN_DIRECTORY = 'train'
_TEST_DIRECTORY = 'test'
_SEGMENTS_FILENAME = 'train_ship_segmentations.csv'
_SAMPLE_SUBMISSION_FILENAME = 'sample_submission.csv'
_BLACKLIST = [
    '13703f040.jpg',
    '14715c06d.jpg',
    '33e0ff2d5.jpg',
    '4d4e09f2a.jpg',
    '877691df8.jpg',
    '8b909bb20.jpg',
    'a8d99130e.jpg',
    'ad55c3143.jpg',
    'c8260c541.jpg',
    'd6c7f17c7.jpg',
    'dc3e7c901.jpg',
    'e44dffe88.jpg',
    'ef87bad36.jpg',
    'f083256d8.jpg']


def read_segments():
    segments_df = pandas.read_csv(_SEGMENTS_FILENAME)

    valid_indices = [_is_valid_image_id(sample.ImageId) for _, sample in segments_df.iterrows()]

    return segments_df[valid_indices]


def read_sample_submission():
    submission_df = pandas.read_csv(_SAMPLE_SUBMISSION_FILENAME)

    valid_indices = [
        _is_valid_image_id(sample.ImageId, test=True) and sample.ImageId not in _BLACKLIST
        for _, sample in submission_df.iterrows()
    ]

    return submission_df[valid_indices]


def sample(dataframe, subsample_size, dropna=True, shuffle=True):
    if dropna:
        dataframe = dataframe.dropna()

    if subsample_size is not None:
        dataframe = dataframe.sample(n=subsample_size)

    if not shuffle:
        return dataframe

    return dataframe.reindex(numpy.random.permutation(dataframe.index))


def generate_x_y(dataframe, test=False):
    for _, value in dataframe.iterrows():
        mask = (
            None if test else
            encoded_pixels_to_mask(value.EncodedPixels, (768, 768))
        )

        yield (
            _image_id_to_array(value.ImageId, test),
            mask,
        )


def generate_batches(samples, batch_size, test=False):
    x_batch = None
    y_batch = None
    batch_idx = 0

    while True:
        for sample_x, sample_y in generate_x_y(samples, test):

            if x_batch is None or y_batch is None:
                x_batch = numpy.array([sample_x] * batch_size)
                y_batch = numpy.array([sample_y] * batch_size)

            if batch_idx == batch_size:
                yield (x_batch, y_batch)
                batch_idx = 0

            x_batch[batch_idx] = sample_x
            y_batch[batch_idx] = sample_y
            batch_idx += 1

        yield (x_batch[0:batch_idx], y_batch[0:batch_idx])


def get_steps_per_epoch(epoch_size, batch_size):
    return int(numpy.ceil(epoch_size / batch_size))


def mask_to_encoded_pixels(mask, threshold=0.5):
    mask = mask.T.flatten()
    indices = numpy.where(mask >= threshold)[0]

    first_index = None
    length = 1
    rle = []
    for index in indices:
        if first_index is None:
            first_index = index
            continue

        if first_index + length == index:
            length += 1
        else:
            rle.extend((first_index + 1, length))
            first_index = index
            length = 1

    if first_index is not None:
        rle.extend((first_index + 1, length))

    return ' '.join(str(x) for x in rle)


def load_model(filename):
    return keras.models.load_model(filename)


def save_model(model, filename):
    model.save(filename)


def save_submission(submission, filename):
    submission.to_csv(filename, index=False)


def _is_valid_image_id(image_id, test=False):
    filename = _from_image_id_to_filename(image_id, test)

    return os.path.exists(filename)


def _image_id_to_array(image_id, test=False):
    if not _is_valid_image_id(image_id, test):
        return None

    filename = _from_image_id_to_filename(image_id, test)

    image = keras.preprocessing.image.load_img(filename)

    return keras.preprocessing.image.img_to_array(image) / 255


def encoded_pixels_to_mask(encoded_pixels, shape):
    if pandas.isna(encoded_pixels):
        return numpy.zeros(shape)

    rle = [int(x) for x in encoded_pixels.split(' ')]

    indices = list(itertools.chain.from_iterable(
        range(rle[idx] - 1, rle[idx] + rle[idx + 1] - 1) for idx in range(0, len(rle), 2)
    ))

    mask = numpy.zeros(shape[0] * shape[1])
    mask[indices] = 1

    return numpy.reshape(mask, (-1, shape[0])).T


def _from_image_id_to_filename(image_id, test):
    directory = _TEST_DIRECTORY if test else _TRAIN_DIRECTORY

    return os.path.join(directory, image_id)
