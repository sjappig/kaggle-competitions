import itertools
import os.path

import keras.models
import keras.preprocessing
import numpy
import pandas

_MODEL_FILENAME = 'shipmodel.h5'
_TRAIN_DIRECTORY = 'train'
_SEGMENTS_FILENAME = 'train_ship_segmentations.csv'


def read_segments():
    segments_df = pandas.read_csv(_SEGMENTS_FILENAME)

    valid_indices = [_is_valid_image_id(sample.ImageId) for _, sample in segments_df.iterrows()]

    return segments_df[valid_indices]


def sample(df, n=None):
    df = df.dropna()

    if n is not None:
        df = df.sample(n=n)

    return df


def generate_x_y(df):
    for _, sample in df.iterrows():
        yield (
            _image_id_to_array(sample.ImageId),
            _encoded_pixels_to_mask(sample.EncodedPixels, (768, 768)),
        )


def load_model():
    return keras.models.load_model(_MODEL_FILENAME)


def save_model(model):
    model.save(_MODEL_FILENAME)

def _is_valid_image_id(image_id):
    filename = _from_image_id_to_filename(image_id)

    return os.path.exists(filename)


def _image_id_to_array(image_id):
    if not _is_valid_image_id(image_id):
        return None

    filename = _from_image_id_to_filename(image_id)

    image = keras.preprocessing.image.load_img(filename)

    return keras.preprocessing.image.img_to_array(image) / 255


def _encoded_pixels_to_mask(encoded_pixels, shape):
    rle = [int(x) for x in encoded_pixels.split(' ')]

    indices = list(
        itertools.chain.from_iterable(range(rle[idx] - 1, rle[idx] + rle[idx + 1] - 1) for idx in range(0, len(rle), 2))
    )

    mask = numpy.zeros(shape[0] * shape[1])
    mask[indices] = 1

    return numpy.reshape(mask, (-1, shape[0])).T


def _from_image_id_to_filename(image_id):
    return os.path.join(_TRAIN_DIRECTORY, image_id)
