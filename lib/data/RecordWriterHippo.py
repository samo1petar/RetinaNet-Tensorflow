import cv2
import json
import numpy as np
from random import shuffle
import os
import tensorflow as tf
from typing import Dict, Generator, List, Union

from lib.tools.progress_bar import printProgressBar
from lib.data.hippo_classes import cls_ids


class RecordWriterHippo:
    def __init__(
            self,
            data_path           : str,
            record_dir          : str,
            record_name         : str,
            annotations         : str,
            save_n_test_images  : int,
            save_n_train_images : int,
    ):
        assert os.path.exists(record_dir)
        assert record_name

        with open(annotations) as f:
            self._annotations = json.load(f)

        self._record_dir  = record_dir
        self._record_name = record_name

        self._train_record = os.path.join(self._record_dir, self._record_name + '_train' + '.tfrecord')
        self._test_record  = os.path.join(self._record_dir, self._record_name + '_test' + '.tfrecord')

        keys = list(self._annotations.keys())
        shuffle(keys)
        train_keys = keys[:int(0.8 * len(keys))]
        test_keys  = keys[int(0.8 * len(keys)):]

        train_dataset = {key: value for key, value in self._annotations.items() if key in train_keys}
        test_dataset = {key: value for key, value in self._annotations.items() if key in test_keys}

        if not os.path.exists(self._test_record):
            self.create_record(test_dataset, self._test_record, save_n_test_images)

        if not os.path.exists(self._train_record):
            self.create_record(train_dataset, self._train_record, save_n_train_images)

    def get_next(self, dataset: Dict, max) -> Generator[List[Union[str, str, bytes]], None, None]:
        for i, (key, annotation) in enumerate(dataset.items()):
            if max is not None and i >= max:
                break

            image = cv2.imread(annotation['path'])

            if image.shape[1] > 900:
                image = cv2.resize(image, (900, int(900 * image.shape[0] / image.shape[1])))

            image = cv2.imencode('.png', image)[1].tostring()

            quads = np.array(annotation['bbox']).reshape(-1, 4, 2)

            bbox_x1 = quads[:, :, 0].min(axis=-1).tolist()
            bbox_y1 = quads[:, :, 1].min(axis=-1).tolist()
            bbox_x2 = quads[:, :, 0].max(axis=-1).tolist()
            bbox_y2 = quads[:, :, 1].max(axis=-1).tolist()

            if isinstance(annotation['class'], str):
                annotation['class'] = [annotation['class']]

            class_ids = np.array([cls_ids[cls] for cls in annotation['class']]).astype(np.int64).tolist()
            yield i, class_ids, bbox_x1, bbox_y1, bbox_x2, bbox_y2, image

    def create_record(self, dataset: Dict, full_record_name: str, max: int) -> None:

        print ('Creating record {}'.format(full_record_name))

        def write(
                index     : int,
                class_ids : List[int],
                bbox_x1   : List[float],
                bbox_y1   : List[float],
                bbox_x2   : List[float],
                bbox_y2   : List[float],
                image     : bytes,
                writer    : tf.compat.v1.python_io.TFRecordWriter,
        ) -> None:
            feature = {
                'index'     : tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'class_ids' : tf.train.Feature(int64_list=tf.train.Int64List(value=class_ids)),
                'bbox_x1'   : tf.train.Feature(float_list=tf.train.FloatList(value=bbox_x1)),
                'bbox_y1'   : tf.train.Feature(float_list=tf.train.FloatList(value=bbox_y1)),
                'bbox_x2'   : tf.train.Feature(float_list=tf.train.FloatList(value=bbox_x2)),
                'bbox_y2'   : tf.train.Feature(float_list=tf.train.FloatList(value=bbox_y2)),
                'image'     : tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
        try:
            with tf.compat.v1.python_io.TFRecordWriter(full_record_name) as writer:
                count = 0
                num_iterate = len(dataset) if max is None else max
                for i, class_ids, bbox_x1, bbox_y1, bbox_x2, bbox_y2, image in self.get_next(dataset, max):
                    count += 1
                    write(i, class_ids, bbox_x1, bbox_y1, bbox_x2, bbox_y2, image, writer)
                    printProgressBar(count, num_iterate, decimals=1, length=50, suffix=' {} / {}'.format(count, num_iterate))

        except Exception as e:
            print ('Writing record failed, erasing record file {}'.format(full_record_name))
            print ('Erorr {}'.format(e))
            os.remove(full_record_name)
