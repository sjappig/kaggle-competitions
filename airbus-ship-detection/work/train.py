import argparse
import importlib

from . import common


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--samples', type=int, default=None, help='number of samples; if not given, use all')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--val', type=float, default=0.2, help='fraction of samples to use for validation')
    parser.add_argument('--queue', type=int, default=10, help='max size of queue used for prefetching batches')
    parser.add_argument('--model', type=str, default='shallow_deconv', help='model to use from work.models-package')
    parser.add_argument('--trained_model', type=str, default=common.MODEL_FILENAME, help='trained model filename')
    parser.add_argument(
        '--save_interval', type=int, metavar='N', default=None, help='do additional saves of model after each N epochs'
    )

    return parser.parse_args()


def read_samples(subsample_size):
    segments_df = common.read_segments()

    print('{} entries available'.format(len(segments_df)), end=' ')
    print('of which {} has ships'.format(segments_df.EncodedPixels.count()))

    return common.sample(segments_df, subsample_size=subsample_size)


def main():
    args = parse_args()

    model_name = 'work.models.{}'.format(args.model)
    print('Using model from {}'.format(model_name))
    model_module = importlib.import_module(model_name)

    model = model_module.create((768, 768, 3))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adadelta',
        metrics=['accuracy', 'mae'],
    )

    model.summary()

    samples = read_samples(args.samples)
    sample_count = len(samples)

    val_len = int(args.val * sample_count)
    train_len = sample_count - val_len

    train_data = samples.iloc[:train_len]
    val_data = samples.iloc[train_len:]

    batch_size = args.batch
    steps_per_epoch = common.get_steps_per_epoch(train_len, batch_size)
    epochs = args.epochs
    max_queue_size = args.queue

    print('sample count: {}'.format(sample_count))
    print('train length: {}'.format(train_len))
    print('validation length: {}'.format(val_len))
    print('batch size: {}'.format(batch_size))
    print('steps per epoch: {}'.format(steps_per_epoch))
    print('epochs: {}'.format(epochs))
    print('max queue size: {}'.format(max_queue_size))

    while epochs > 0:
        sub_epochs = epochs if args.save_interval is None else min(args.save_interval, epochs)

        model.fit_generator(
            common.generate_batches(train_data, batch_size),
            steps_per_epoch=steps_per_epoch,
            epochs=sub_epochs,
            max_queue_size=max_queue_size,
            validation_data=common.generate_batches(val_data, batch_size),
            validation_steps=common.get_steps_per_epoch(val_len, batch_size),
        )

        print('Saving model... ', end='')
        common.save_model(model, args.trained_model)
        print('done')

        epochs -= sub_epochs


if __name__ == '__main__':
    main()
