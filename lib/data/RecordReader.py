import os
import tensorflow as tf
from typing import Generator

from lib.image import resize_and_pad_image
from lib.bbox import convert_to_xywh
from lib.LabelEncoder import LabelEncoder


class RecordReader:
    """
    Here is where the record, created by RecordWriter, is being read and where the preprocessing is done.
    Params:
    record_dir          : string -> directory where you want to save records
    record_name         : string -> record name
    label_encoder       : LabelEncoder -> need for creating targets
    rotate              : boolean -> use random rotation in preprocessing
    batch_size          : integer -> size of a single batch
    shuffle_buffer      : integer -> buffer size from which images will be randomly picked for the next batch
    num_parallel_calls  : integer -> number of processes to run preprocessing scripts in parallel
    num_parallel_reads  : integer -> number of processes to run reading the record in parallel
    prefetch_buffer_size: integer -> read data in advanced
    count               : integer -> how many images will be read. If None, loop without stopping

    All these parameters are described in https://www.tensorflow.org/guide/data_performance
    """
    def __init__(
            self,
            record_dir           : str,
            record_name          : str,
            label_encoder        : LabelEncoder,
            rotate               : bool = False,
            batch_size           : int = 1,
            shuffle_buffer       : int = 100,
            num_parallel_calls   : int = 8,
            num_parallel_reads   : int = 32,
            prefetch_buffer_size : int = 100,
            count                : int = None,
    ):
        assert os.path.exists(record_dir)
        assert record_name

        self._record_dir           = record_dir
        self._record_name          = record_name
        self._label_encoder        = label_encoder
        self._rotate               = rotate
        self._batch_size           = batch_size
        self._shuffle_buffer       = shuffle_buffer
        self._num_parallel_calls   = num_parallel_calls
        self._num_parallel_reads   = num_parallel_reads
        self._prefetch_buffer_size = prefetch_buffer_size
        self._count                = count

    def read_record(self, name: str, create_labels: bool = True) -> Generator[tf.Tensor, None, None]:
        # check if name is 'train' or 'test'
        assert name in ['train', 'test']
        # get full record name
        full_record_name = os.path.join(self._record_dir, self._record_name + '_' + name + '.tfrecord')
        # function that knows how to extract data from the record. Must be compatible with
        # RecirdWriter write() function that's inside create_record() method
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
        # process is preprocessing function
        def process(index, class_ids, bbox_x1, bbox_y1, bbox_x2, bbox_y2, image):
            # resize the image so it goes through piramid style neural network
            image, image_shape, ratio = resize_and_pad_image(image, min_side=300, max_side=800, jitter=None)
            # normalize the image so it's in [0, 1] range
            image = image / 255
            # add new axis in front of bounding boxes lists
            bbox_x1 = bbox_x1[tf.newaxis, ...]
            bbox_y1 = bbox_y1[tf.newaxis, ...]
            bbox_x2 = bbox_x2[tf.newaxis, ...]
            bbox_y2 = bbox_y2[tf.newaxis, ...]
            # merge bounding boxes
            bbox = tf.concat(
                (tf.transpose(bbox_x1), tf.transpose(bbox_y1),
                 tf.transpose(bbox_x2), tf.transpose(bbox_y2)),
                axis=1,
            )
            # resize bounding boxes the same as the images
            bbox = tf.stack(
                [
                    bbox[:, 0] * ratio,
                    bbox[:, 1] * ratio,
                    bbox[:, 2] * ratio,
                    bbox[:, 3] * ratio,
                ],
                axis=-1,
            )
            if self._rotate:
                # function for rotating bounding boxes by 90 degrees
                def rot_bbox(bbox, height):
                    return tf.stack((
                        tf.cast(height, tf.float32) - bbox[..., 3], bbox[..., 0],
                        tf.cast(height, tf.float32) - bbox[..., 1], bbox[..., 2],
                    ), axis=-1)
                # random flag for rotation decision
                rot = tf.random.uniform([], 0, 4, dtype=tf.int32)
                # new bounding boxes, maybe rotated
                bbox = tf.switch_case(
                    rot, branch_fns={
                        0: lambda: bbox,
                        1: lambda: rot_bbox(rot_bbox(rot_bbox(bbox, tf.shape(image)[0]), tf.shape(image)[1]), tf.shape(image)[0]),
                        2: lambda: rot_bbox(rot_bbox(bbox, tf.shape(image)[0]), tf.shape(image)[1]),
                        3: lambda: rot_bbox(bbox, tf.shape(image)[0]),
                    }, default=None,
                )
                # rotate image in the same way as the bounding boxes
                image = tf.image.rot90(image, k=rot)
            # convert bounding boxes from [x1, y1, x2, y2] to [x, y, width, height] format
            bbox = convert_to_xywh(bbox)
            # cast the class indexes to int32 type
            class_ids = tf.cast(class_ids, dtype=tf.int32)
            # return data
            return image, bbox, class_ids

        def augment(image, bbox, class_ids):
            """
            for image augmentation techniques -> brightness, hue, constrast and saturation
            """
            image_2 = tf.image.random_brightness(image, 0.2)
            image_2 = tf.image.random_hue(image_2, 0.2)
            image_2 = tf.image.random_contrast(image_2, 0.0, 0.3)
            image_2 = tf.image.random_saturation(image_2, 2, 10)

            image = tf.case([(tf.cast(tf.random.uniform([], 0, 2, dtype=tf.int32), dtype=tf.bool), lambda: image)],
                default=lambda: image_2,
            )

            return image, bbox, class_ids

        if name == 'test':
            # if test is used, use only 1 only once go through the dataset and use batch size 1
            batch_size = 1
            count = 1
        else:
            # else train -> use batch size as it is and loop over the dataset over and over again
            batch_size = self._batch_size
            count = self._count

        # record reader -> patching all the elements together
        # https://www.tensorflow.org/guide/data_performance
        autotune = tf.data.experimental.AUTOTUNE

        dataset = tf.data.TFRecordDataset(full_record_name, num_parallel_reads=self._num_parallel_reads)
        dataset = dataset.map(parse, num_parallel_calls=self._num_parallel_calls)
        dataset = dataset.map(process, num_parallel_calls=self._num_parallel_calls)
        if name == 'train':
            # use image augmentation only for train
            dataset = dataset.map(augment, num_parallel_calls=self._num_parallel_calls)
        dataset = dataset.shuffle(self._shuffle_buffer)
        dataset = dataset.padded_batch(batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True)
        if create_labels:
            dataset = dataset.map(self._label_encoder.encode_batch, num_parallel_calls=autotune)
        dataset = dataset.apply(tf.data.experimental.ignore_errors())

        dataset = dataset.repeat(count=count)
        dataset = dataset.prefetch(buffer_size=self._prefetch_buffer_size)

        return dataset
