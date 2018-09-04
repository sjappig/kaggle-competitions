import itertools
import os.path

from keras.applications.vgg16 import VGG16
from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Dense, Flatten, MaxPooling2D, Reshape
from keras.models import Model, Sequential, load_model
from keras.preprocessing import image
import matplotlib.pyplot as plot
import numpy
import pandas
import sklearn.model_selection

MODEL_FILENAME='shipmodel.h5'

_TRAIN_DIRECTORY = 'train'
_SEGMENTS_FILENAME = 'train_ship_segmentations.csv'

def is_valid_image_id(image_id):
    filename = os.path.join(_TRAIN_DIRECTORY, image_id)
    return os.path.exists(filename)

def image_id_to_array(image_id):
    if not is_valid_image_id(image_id):
        return None

    filename = os.path.join(_TRAIN_DIRECTORY, image_id)

    img = image.load_img(filename)
    return image.img_to_array(img) / 255

def encoded_pixels_to_mask(encoded_pixels, shape):
    rle = [int(x) for x in encoded_pixels.split(' ')]
    indices = list(
        itertools.chain.from_iterable(range(rle[idx] - 1, rle[idx] + rle[idx + 1] - 1) for idx in range(0, len(rle), 2))
    )
    mask = numpy.zeros(shape[0] * shape[1])
    mask[indices] = 1
    return numpy.reshape(mask, (-1, shape[0])).T

if __name__ == '__main__':
    segments_df = pandas.read_csv(_SEGMENTS_FILENAME)
    print('Total {} entries'.format(len(segments_df)))

    valid_indices = [is_valid_image_id(sample.ImageId) for _, sample in segments_df.iterrows()]
    segments_df = segments_df[valid_indices]

    print('{} entries available'.format(len(segments_df)), end=' ')
    print('of which {} has ships'.format(segments_df.EncodedPixels.count()))

    segments_df = segments_df.dropna()

    samples = segments_df.sample(n=16)

    x = [image_id_to_array(sample.ImageId) for _, sample in samples.iterrows()]
    y = [encoded_pixels_to_mask(sample.EncodedPixels, (768, 768)) for _, sample in samples.iterrows()]

    #x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, stratify=y)

    x_train = numpy.array(x)
    y_train = numpy.array(y)

    print('Input shape: {}'.format(x_train.shape))
    print('Output shape: {}'.format(y_train.shape))

    all_pixels = len(y_train) * 768 * 768
    ship_pixels = numpy.sum(y_train)
    blank_pixels = all_pixels - ship_pixels

    print('{} pixels, {} has ship, {} do not'.format(all_pixels, ship_pixels, blank_pixels))

    model = load_model(MODEL_FILENAME)

    for idx in range(0, 3):
        mask = model.predict(x_train[idx:idx+1])[0]
        img = x_train[idx]
        truth = y_train[idx]
        plot.subplot(3, 3, idx*3 + 1)
        plot.imshow(img)
        plot.subplot(3, 3, idx*3 + 2)
        plot.imshow(mask)
        plot.subplot(3, 3, idx*3 + 3)
        plot.imshow(truth)

    plot.show()
