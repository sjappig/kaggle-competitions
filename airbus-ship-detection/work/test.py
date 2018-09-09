import argparse
import threading
import queue
import sys

from . import common


def parse_args():
    parser = argparse.ArgumentParser(description='Test segmentation network')

    parser.add_argument('--samples', type=int, default=None, help='number of samples; if not given, use all')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--queue', type=int, default=10, help='max size of queue used for prefetching batches')
    parser.add_argument('--trained_model', type=str, default=common.MODEL_FILENAME, help='trained model filename')
    parser.add_argument('--submission', type=str, default=common.SUBMISSION_FILENAME, help='submission filename')

    return parser.parse_args()


def generate_predictions(model, batch_queue):
    while True:
        batch = batch_queue.get()
        batch_queue.task_done()
        if batch is None:
            break
        predictions = model.predict_on_batch(batch)

        yield from predictions

        print('.', end='')
        sys.stdout.flush()


def main():
    args = parse_args()

    model = common.load_model(args.trained_model)

    sample_submission_df = common.read_sample_submission()

    print('{} entries available'.format(len(sample_submission_df)))

    samples = common.sample(sample_submission_df, subsample_size=args.samples, dropna=False, shuffle=False)
    sample_count = len(samples)

    batch_size = args.batch
    steps_per_epoch = common.get_steps_per_epoch(sample_count, batch_size)
    max_queue_size = args.queue

    print('sample count: {}'.format(sample_count))
    print('batch size: {}'.format(batch_size))
    print('steps per epoch: {}'.format(steps_per_epoch))
    print('max queue size: {}'.format(max_queue_size))

    submission = samples.copy()
    batch_queue = queue.Queue(max_queue_size)

    def feed_queue():
        step = 0

        for x_batch, _ in common.generate_batches(samples, batch_size, test=True):
            batch_queue.put(x_batch)
            step += 1
            if step == steps_per_epoch:
                break

        batch_queue.put(None)

    worker = threading.Thread(target=feed_queue)
    worker.start()

    ctr = 0
    for prediction in generate_predictions(model, batch_queue):
        submission.iloc[ctr].EncodedPixels = common.mask_to_encoded_pixels(prediction)
        ctr += 1

    print('done')
    batch_queue.join()
    worker.join()

    common.save_submission(submission, args.submission)


if __name__ == '__main__':
    main()
