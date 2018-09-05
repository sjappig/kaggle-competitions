import matplotlib.pyplot as plot
import numpy

from . import common

if __name__ == '__main__':
    segments_df = common.read_segments()

    samples = common.sample(segments_df, n=3)

    x, y = zip(*common.generate_x_y(samples))

    x_train = numpy.array(x)
    y_train = numpy.array(y)

    model = common.load_model()

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
