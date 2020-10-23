import os
import tensorflow as tf
from typing import Generator

from lib.image import resize_and_pad_image
from lib.bbox import convert_to_xywh
from lib.LabelEncoder import LabelEncoder

class RecordReader:
    def __init__(
            self,
            record_dir           : str,
            record_name          : str,
            label_encoder        : LabelEncoder,
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
        self._label_encoder        = label_encoder
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

        def process(index, class_ids, bbox_x1, bbox_y1, bbox_x2, bbox_y2, image):

            image, image_shape, ratio = resize_and_pad_image(image, min_side=300, max_side=800, jitter=None)
            image = image / 255

            bbox_x1 = bbox_x1[tf.newaxis, ...]
            bbox_y1 = bbox_y1[tf.newaxis, ...]
            bbox_x2 = bbox_x2[tf.newaxis, ...]
            bbox_y2 = bbox_y2[tf.newaxis, ...]

            bbox = tf.concat(
                (tf.transpose(bbox_x1), tf.transpose(bbox_y1),
                 tf.transpose(bbox_x2), tf.transpose(bbox_y2)),
                axis=1,
            )
            bbox = tf.stack(
                [
                    bbox[:, 0] * ratio,
                    bbox[:, 1] * ratio,
                    bbox[:, 2] * ratio,
                    bbox[:, 3] * ratio,
                ],
                axis=-1,
            )
            bbox = convert_to_xywh(bbox)
            class_ids = tf.cast(class_ids, dtype=tf.int32)

            return image, bbox, class_ids

        def augment(image, bbox, class_ids):

            image_2 = tf.image.random_brightness(image, 0.2)
            image_2 = tf.image.random_hue(image_2, 0.2)
            image_2 = tf.image.random_contrast(image_2, 0.0, 0.3)
            image_2 = tf.image.random_saturation(image_2, 2, 10)

            image = tf.cond(
                tf.random.uniform([], 0, 2, dtype=tf.int32),
                true_fn=lambda: image,
                false_fn=lambda: image_2,
            )

            return image, bbox, class_ids

        if name == 'test':
            batch_size = 1
            count = 1
        else:
            batch_size = self._batch_size
            count = self._count

        autotune = tf.data.experimental.AUTOTUNE

        dataset = tf.data.TFRecordDataset(full_record_name, num_parallel_reads=self._num_parallel_reads)
        dataset = dataset.map(parse, num_parallel_calls=self._num_parallel_calls)
        dataset = dataset.map(process, num_parallel_calls=self._num_parallel_calls)
        # if name == 'train':
        #     dataset = dataset.map(augment, num_parallel_calls=self._num_parallel_calls)
        dataset = dataset.shuffle(self._shuffle_buffer)
        dataset = dataset.padded_batch(batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True)
        dataset = dataset.map(self._label_encoder.encode_batch, num_parallel_calls=autotune)
        dataset = dataset.apply(tf.data.experimental.ignore_errors())

        dataset = dataset.repeat(count=count)
        dataset = dataset.prefetch(buffer_size=self._prefatch_buffer_size)

        return dataset

        # for images, labels in dataset:
        #     yield images, labels


        # for index, class_ids, bbox_x1, bbox_y1, bbox_x2, bbox_y2, image in iter(dataset):
        #
        #     # B = image[..., 0]
        #     # G = image[..., 1]
        #     # R = image[..., 2]
        #     #
        #     # image = tf.stack((R, G, B), axis=-1)
        #
        #     image, image_shape, ratio = resize_and_pad_image(image[0], min_side=300, max_side=800, jitter=None)
        #
        #     image = image / 255
        #
        #     image = tf.expand_dims(image, axis=0)
        #
        #     bbox = tf.concat(
        #         (tf.transpose(bbox_x1), tf.transpose(bbox_y1),
        #          tf.transpose(bbox_x2), tf.transpose(bbox_y2)),
        #         axis=1,
        #     )
        #
        #     bbox = tf.stack(
        #         [
        #             bbox[:, 0] * ratio,
        #             bbox[:, 1] * ratio,
        #             bbox[:, 2] * ratio,
        #             bbox[:, 3] * ratio,
        #         ],
        #         axis=-1,
        #     )
        #     bbox = convert_to_xywh(bbox)
        #
        #     class_ids = class_ids[0]
        #
        #
        #     # ############################################
        #     # def show(image):
        #     #     import cv2
        #     #     cv2.imshow('', image)
        #     #     cv2.waitKey()
        #     #     cv2.destroyAllWindows()
        #     #
        #     # bbox_np = bbox.numpy()
        #     #
        #     # import numpy as np
        #     # import cv2
        #     # image_np = (image[0].numpy() * 255).astype(np.uint8)
        #     #
        #     # isClosed = True
        #     # color = (255, 0, 0)
        #     # thickness = 2
        #     # for x, y, w, h in bbox_np:
        #     #     # points = np.array([
        #     #     #     [x1, y1],
        #     #     #     [x1, y2],
        #     #     #     [x2, y2],
        #     #     #     [x2, y1],
        #     #     # ], np.int32).reshape((-1, 1, 2))
        #     #     points = np.array([
        #     #         [x   - w/2, y   - h/2],
        #     #         [x+w - w/2, y   - h/2],
        #     #         [x+w - w/2, y+h - h/2],
        #     #         [x   - w/2, y+h - h/2],
        #     #     ], np.int32).reshape((-1, 1, 2))
        #     #     image_np = cv2.polylines(image_np, [points], isClosed, color, thickness)
        #     #
        #     # print ('Record Reader')
        #     # from IPython import embed
        #     # embed()
        #     # exit()
        #     # ############################################
        #
        #     # yield index, class_ids, bbox, image
        #
        #     yield image, bbox, class_ids
