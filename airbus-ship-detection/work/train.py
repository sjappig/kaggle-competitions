import itertools
import os.path

from keras.applications.vgg16 import VGG16
from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Dense, Flatten, MaxPooling2D, Reshape
from keras.models import Model, Sequential
from keras.preprocessing import image
import numpy
import pandas
import sklearn.model_selection

MODEL_FILENAME='shipmodel.h5'

_TRAIN_DIRECTORY = 'train'
_SEGMENTS_FILENAME = 'train_ship_segmentations.csv'

def get_feature_extraction_model(input_shape, model_name='adhoc'):
    if model_name == 'vgg16':
        model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

        # allow retraining of the last convolutional layers
        model.layers[-2].trainable = True
        model.layers[-3].trainable = True
        model.layers[-4].trainable = True

        return model

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

    samples = segments_df.sample(n=2*1024)

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

    feat_model = get_feature_extraction_model((768, 768, 3))
    output = Conv2DTranspose(filters=3, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(feat_model.output)
    output = BatchNormalization()(output)
    output = Conv2DTranspose(filters=3, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(output)
    output = BatchNormalization()(output)
    output = Conv2DTranspose(filters=3, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(output)
    output = BatchNormalization()(output)
    output = Conv2DTranspose(filters=1, kernel_size=3, strides=(2, 2), padding='same', activation='sigmoid')(output)
    output = Reshape((768, 768))(output)
    # output = Flatten()(feat_model.output)
    # output = Dense(128, activation='relu')(output)
    # output = Dense(1, activation='sigmoid')(output)

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
    #    class_weight={ 0: blank_pixels, 1: ship_pixels },
    )

    model.save(MODEL_FILENAME)

