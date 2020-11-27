import cv2
import json
import numpy as np
from random import shuffle
import os
import tensorflow as tf
from typing import Dict, Generator, List, Union

from lib.tools.progress_bar import printProgressBar
from lib.data.hippo_classes import cls_ids


class RecordWriter:
    """
    Write record with all the data. The record is a file with all the images and annotations in such a format that
    tensorflow likes. In that way, training is extra fast. No need for image loading and preprocessing.
    RecordWriterHippo creates that records. It creates 80:20 split for train and test sets.
    Creating records takes time, so it's only done once.  Once the records are created it won't create another one
    with the same name.
    Params:
    record_dir          : string -> directory where you want to save records
    record_name         : string -> record name
    annotations         : string -> path to file with annotations TODO [example link]
    save_n_test_images  : integer -> up to how many images will be saved in test set, if None -> use all the images
    save_n_train_images : integer -> up to how many images will be saved in train set, if None -> use all the images
    """
    def __init__(
            self,
            record_dir          : str,
            record_name         : str,
            annotations         : str,
            save_n_test_images  : int,
            save_n_train_images : int,
    ):
        # check if record_dir exists
        assert os.path.exists(record_dir)
        # load annotations
        with open(annotations) as f:
            self._annotations = json.load(f)

        self._record_dir  = record_dir
        self._record_name = record_name
        # create train and test record full paths
        self._train_record = os.path.join(self._record_dir, self._record_name + '_train' + '.tfrecord')
        self._test_record  = os.path.join(self._record_dir, self._record_name + '_test' + '.tfrecord')
        # get data keys. Keys uniquely identify the images
        keys = list(self._annotations.keys())
        # shuffle the keys
        shuffle(keys)
        # creaate train-test key-only split 80:20
        train_keys = keys[:int(0.8 * len(keys))]
        test_keys  = keys[int(0.8 * len(keys)):]
        # create train-test split with all data from annotations
        train_dataset = {key: value for key, value in self._annotations.items() if key in train_keys}
        test_dataset = {key: value for key, value in self._annotations.items() if key in test_keys}
        # if test record not created -> create it
        if not os.path.exists(self._test_record):
            self.create_record(test_dataset, self._test_record, save_n_test_images)
        # if train record not created -> create it
        if not os.path.exists(self._train_record):
            self.create_record(train_dataset, self._train_record, save_n_train_images)

    def get_next(self, dataset: Dict, max) -> Generator[List[Union[str, str, bytes]], None, None]:
        """
        get_next is being called from create_record(). It's a function where the data is being loaded.
        Image is loaded here, bounding boxes are loaded here and some preprocessing is also done here.
        Once the data gets to the record, it's fixed there. Meaning, if you want to remove certain image or
        change some small thing, you can't. You need to recreate the record.
        """
        for i, (key, annotation) in enumerate(dataset.items()):
            if max is not None and i >= max:
                break
            # read the image
            image = cv2.imread(annotation['path'][0])
            # resize. This is important to do because annotations won't fit if not resized exactly like this
            if image.shape[1] > 900:
                image = cv2.resize(image, (900, int(900 * image.shape[0] / image.shape[1])))
            # encode image to string. This is how tensorflow writer can save an image to the record
            image = cv2.imencode('.png', image)[1].tostring()
            # get bounding boxes
            quads = np.array(annotation['bbox']).reshape(-1, 4, 2)
            # split bounding boxes and save to lists
            bbox_x1 = quads[:, :, 0].min(axis=-1).tolist()
            bbox_y1 = quads[:, :, 1].min(axis=-1).tolist()
            bbox_x2 = quads[:, :, 0].max(axis=-1).tolist()
            bbox_y2 = quads[:, :, 1].max(axis=-1).tolist()
            # save classes to the list
            if isinstance(annotation['class'], str):
                annotation['class'] = [annotation['class']]
            # get integer representations of the classes
            class_ids = np.array([cls_ids[cls] for cls in annotation['class']]).astype(np.int64).tolist()
            # yielf the results and go get another one
            yield i, class_ids, bbox_x1, bbox_y1, bbox_x2, bbox_y2, image

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
