import os
import tensorflow as tf
from typing import Generator


class RecordReader:
    def __init__(
            self,
            record_dir           : str = None,
            record_name          : str = None,
            batch_size           : int = 1,
            shuffle_buffer       : int = 100,
            num_parallel_calls   : int = 8,
            num_parallel_reads   : int = 32,
            prefatch_buffer_size : int = 100,
            count                : int = None,
    ):
        assert os.path.exists(record_dir)
        assert record_name

        self._record_dir           = record_dir
        self._record_name          = record_name
        self._batch_size           = batch_size
        self._shuffle_buffer       = shuffle_buffer
        self._num_parallel_calls   = num_parallel_calls
        self._num_parallel_reads   = num_parallel_reads
        self._prefatch_buffer_size = prefatch_buffer_size
        self._count                = count

    def read_record(self, name: str) -> Generator[tf.Tensor, None, None]:

        assert name in ['train', 'test']

        full_record_name = os.path.join(self._record_dir, self._record_name + '_' + name + '.tfrecord')

        def parse(x):
            keys_to_features = {
                'index'     : tf.io.FixedLenFeature([], tf.int64),
                'class_ids' : tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                'bbox_x1'   : tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
                'bbox_y1'   : tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
                'bbox_x2'   : tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
                'bbox_y2'   : tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
                'image'     : tf.io.FixedLenFeature([], tf.string),
            }
            parsed_features = tf.io.parse_single_example(x, keys_to_features)
            parsed_features['image'] = tf.image.decode_png(parsed_features['image'], channels=3)
            return (
                parsed_features['index'],
                parsed_features['class_ids'],
                parsed_features['bbox_x1'],
                parsed_features['bbox_y1'],
                parsed_features['bbox_x2'],
                parsed_features['bbox_y2'],
                parsed_features['image'],
            )
        if name == 'test':
            batch_size = 1
            count = 1
        else:
            batch_size = self._batch_size
            count = self._count
        dataset = tf.data.TFRecordDataset(full_record_name, num_parallel_reads=self._num_parallel_reads)
        dataset = dataset.map(parse, num_parallel_calls=self._num_parallel_calls)
        dataset = dataset.batch(batch_size)
        dataset = dataset.shuffle(self._shuffle_buffer)
        dataset = dataset.repeat(count=count)
        dataset = dataset.prefetch(buffer_size=self._prefatch_buffer_size)

        for index, class_ids, bbox_x1, bbox_y1, bbox_x2, bbox_y2, image in iter(dataset):

            B = image[..., 0]
            G = image[..., 1]
            R = image[..., 2]

            image = tf.stack((R, G, B), axis=-1)

            image = tf.cast(image, dtype=tf.float32)
            image = image / 255

            bboxes = tf.concat(
                (tf.transpose(bbox_x1), tf.transpose(bbox_y1),
                 tf.transpose(bbox_x2), tf.transpose(bbox_y2)),
                axis=1,
            )

            class_ids = class_ids[0]

            yield index, class_ids, bboxes, image
