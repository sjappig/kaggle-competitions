import matplotlib.pyplot as plot
import numpy

from . import common


def main():
    segments_df = common.read_segments()

    samples = common.sample(segments_df, subsample_size=3)

    x_samples, y_samples = zip(*common.generate_x_y(samples))

    x_samples = numpy.array(x_samples)
    y_samples = numpy.array(y_samples)

    model = common.load_model()

    for idx in range(0, 3):
        mask = model.predict(x_samples[idx:idx+1])[0]
        img = x_samples[idx]
        truth = y_samples[idx]
        plot.subplot(3, 3, idx*3 + 1)
        plot.imshow(img)
        plot.subplot(3, 3, idx*3 + 2)
        plot.imshow(mask)
        plot.subplot(3, 3, idx*3 + 3)
        plot.imshow(truth)

    plot.show()


if __name__ == '__main__':
    main()
