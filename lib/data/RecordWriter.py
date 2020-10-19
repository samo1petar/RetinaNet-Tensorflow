import cv2
from gluoncv import data
import numpy as np
import os
import tensorflow as tf
from typing import Generator, List, Union

from lib.tools.progress_bar import printProgressBar


class RecordWriter:
    def __init__(
            self,
            data_path           : str = '/media/david/A/Dataset/COCO',  # TODO change data path
            record_dir          : str = None,
            record_name         : str = None,
            save_n_test_images  : int = None,
            save_n_train_images : int = None,
    ):
        assert os.path.exists(record_dir)
        assert record_name

        self._record_dir  = record_dir
        self._record_name = record_name

        self._train_record = os.path.join(self._record_dir, self._record_name + '_train' + '.tfrecord')
        self._test_record  = os.path.join(self._record_dir, self._record_name + '_test' + '.tfrecord')

        if not os.path.exists(self._test_record):
            test_dataset = data.COCODetection(root=data_path, splits=['instances_val2017'])
            self.create_record(test_dataset, self._test_record, save_n_test_images)

        if not os.path.exists(self._train_record):
            train_dataset = data.COCODetection(root=data_path, splits=['instances_train2017'])
            self.create_record(train_dataset, self._train_record, save_n_train_images)

    def get_next(self, dataset: data.mscoco.detection.COCODetection, max) -> Generator[List[Union[str, str, bytes]], None, None]:
        for i, (image, label) in enumerate(dataset):
            if max is not None and i >= max:
                break
            image = image.asnumpy()
            image = cv2.imencode('.png', image)[1].tostring()
            bbox_x1 = label[:, 0].tolist()
            bbox_y1 = label[:, 1].tolist()
            bbox_x2 = label[:, 2].tolist()
            bbox_y2 = label[:, 3].tolist()
            class_ids = label[:, 4].astype(np.int64).tolist()
            yield i, class_ids, bbox_x1, bbox_y1, bbox_x2, bbox_y2, image

    def create_record(self, dataset: data.mscoco.detection.COCODetection, full_record_name: str, max: int) -> None:

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
