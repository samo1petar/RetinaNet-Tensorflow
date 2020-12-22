import cv2
import json
import numpy as np
from random import shuffle
import os
import tensorflow as tf
from typing import Dict, Generator, List, Union

from lib.tools.progress_bar import printProgressBar
from lib.data.classes import cls_ids
from lib.tools.annotation import bs_from_xml, objects_from_bs

from IPython import embed # TODO remove

class RecordWriter:

    def __init__(
            self,
            dataset             : str,
            annotations         : str,
            record_dir          : str,
            record_name         : str,
            save_n_test_images  : int,
            save_n_train_images : int,
            train_percentage    : float = 0.8,
    ):
        assert os.path.exists(dataset)
        assert os.path.exists(annotations)

        if not os.path.exists(record_dir):
            os.makedirs(record_dir)

        self._record_dir  = record_dir
        self._record_name = record_name

        self._train_record = os.path.join(self._record_dir, self._record_name + '_train' + '.tfrecord')
        self._test_record  = os.path.join(self._record_dir, self._record_name + '_test' + '.tfrecord')

        keys = {x.rsplit('.', 1)[0] for x in os.listdir(annotations)}
        data = {x.rsplit('.', 1)[0]: {
            'image': os.path.join(dataset, x),
            'annotation': os.path.join(annotations, x.rsplit('.', 1)[0] + '.xml'),
        }  for x in os.listdir(dataset) if x.rsplit('.', 1)[0] in keys}

        data_keys = list(data.keys())

        shuffle(data_keys)

        train_keys = data_keys[:int(train_percentage * len(data_keys))]
        test_keys  = data_keys[int(train_percentage * len(data_keys)):]

        train_dataset = {key: value for key, value in data.items() if key in train_keys}
        test_dataset  = {key: value for key, value in data.items() if key in test_keys}

        if not os.path.exists(self._test_record):
            self.create_record(test_dataset, self._test_record, save_n_test_images)

        if not os.path.exists(self._train_record):
            self.create_record(train_dataset, self._train_record, save_n_train_images)

    def get_next(self, dataset: Dict, max) -> Generator[List[Union[str, str, bytes]], None, None]:
        try:
            for i, (key, annotation) in enumerate(dataset.items()):
                if max is not None and i >= max:
                    break

                if not os.path.exists(annotation['annotation']):
                    yield i, None, None, None, None, None, None

                class_ids = []
                bboxes = []

                for obj in objects_from_bs(bs_from_xml(annotation['annotation'])):
                    class_ids.append(cls_ids[obj['cls']])
                    bboxes.append(obj['bbox'])

                bboxes = np.array(bboxes, dtype=np.float32).reshape(-1, 4)

                image = cv2.imread(annotation['image'])

                if image.shape[1] > 600:
                    resize_ratio = 600. / image.shape[1]
                    image = cv2.resize(image, (600, int(600 * image.shape[0] / image.shape[1])))

                    bboxes *= resize_ratio

                bbox_x1 = bboxes[:, 0].tolist()
                bbox_y1 = bboxes[:, 1].tolist()
                bbox_x2 = bboxes[:, 2].tolist()
                bbox_y2 = bboxes[:, 3].tolist()

                image = cv2.imencode('.png', image)[1].tostring()

                yield i, class_ids, bbox_x1, bbox_y1, bbox_x2, bbox_y2, image
        except Exception as e:
            embed()
            exit()

    def create_record(self, dataset: Dict, full_record_name: str, max: int) -> None:
        """
        Here is the record beeing created.
        Tensorflow has a specific way to create records, and it's a bit messy but that's how they did it.
        record_writer is creted with writer=tf.compat.v1.python_io.TFRecordWriter(full_record_name) and then
        by calling writer.write(...) record is being writen to.
        """
        print ('Creating record {}'.format(full_record_name))

        # write() function describes how to write data to a record. This is record input and RecordReader must know
        # how to read it. RecordReader must have complimentary function with this one. It's called parse() and it's
        # inside RecordReader.read_record() -> parse()
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
        # safeguard the record creation
        try:
            # create record writer
            with tf.compat.v1.python_io.TFRecordWriter(full_record_name) as writer:
                # count  and max_iter are for speed feedback
                count = 0
                num_iterate = len(dataset) if max is None else max
                # start to loop through data
                for i, class_ids, bbox_x1, bbox_y1, bbox_x2, bbox_y2, image in self.get_next(dataset, max):
                    if image == None:
                        continue
                    count += 1
                    # write the image and it's annotations to the record
                    write(i, class_ids, bbox_x1, bbox_y1, bbox_x2, bbox_y2, image, writer)
                    # alert the user for progress
                    printProgressBar(count, num_iterate, decimals=1, length=50, suffix=' {} / {}'.format(count, num_iterate))
        # if error occures erase the record and print the error
        except Exception as e:
            print ('Writing record failed, erasing record file {}'.format(full_record_name))
            print ('Erorr {}'.format(e))
            # remove the record. Unfinished record is of no use, and if not erased, RecordWriter and RecordReader
            # will see the record and think it's done properly. Must erase it.
            os.remove(full_record_name)
