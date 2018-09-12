import argparse

import matplotlib.pyplot as plot
import numpy

from . import common


def parse_args():
    parser = argparse.ArgumentParser(description='View samples with their correct segmentations and predictions')

    parser.add_argument('--trained_model', type=str, default=common.MODEL_FILENAME, help='trained model filename')
    parser.add_argument(
        '--threshold', type=float, default=0.5, help='threshold when transforming probablity map to binary mask'
    )
    parser.add_argument('--ignore-empty', dest='dropna', action='store_true', help='ignore images without ships')

    parser.add_argument('--samples', type=int, default=4, help='number of samples to view')

    return parser.parse_args()


def main():
    args = parse_args()

    segments_df = common.read_segments()

    samples_to_view = args.samples
    dropna = True if args.dropna else False

    samples = common.sample(segments_df, dropna=dropna, subsample_size=samples_to_view)

    x_samples, y_samples = zip(*common.generate_x_y(samples))

    x_samples = numpy.array(x_samples)
    y_samples = numpy.array(y_samples)

    model = common.load_model(args.trained_model)

    for idx in range(0, len(samples)):
        mask = model.predict(x_samples[idx:idx+1])[0]
        mask[numpy.where(mask > args.threshold)] = 1
        mask[numpy.where(mask != 1)] = 0
        img = x_samples[idx]
        truth = y_samples[idx]
        plot.subplot(samples_to_view, 3, idx*3 + 1)
        plot.imshow(img)
        plot.subplot(samples_to_view, 3, idx*3 + 2)
        plot.imshow(mask)
        plot.subplot(samples_to_view, 3, idx*3 + 3)
        plot.imshow(truth)

    plot.show()


if __name__ == '__main__':
    main()
