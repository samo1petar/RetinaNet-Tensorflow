import argparse
import numpy as np
import tensorflow as tf

from lib.data.hippo_classes import cls_ids
from lib.data.RecordReader import RecordReader
from lib.LabelEncoder import LabelEncoder
from lib.layers.DecodePredictions import DecodePredictions
from lib.visualize import visualize_detections

from params import Params as p

'''
This script is for visualizing the results.
'''

def inference(
        model_dir: str,
        split: str,
        record_dir: str,
        record_name: str,
        number_of_images_shown: int,
):
    model = p.model

    latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    model.load_weights(latest_checkpoint)

    decoder = DecodePredictions(confidence_threshold=0.5)

    record_reader = RecordReader(
            record_dir           = record_dir,
            record_name          = record_name,
            label_encoder        = LabelEncoder(),
            batch_size           = 1,
            shuffle_buffer       = 1,
            num_parallel_calls   = 1,
            num_parallel_reads   = 1,
            prefetch_buffer_size = 1,
        )

    dataset = record_reader.read_record(split, create_labels=True)

    i = 0

    for input_image, _ in dataset:
        if i >= number_of_images_shown: break
        i += 1
        input_image = tf.cast(input_image, dtype=tf.float32)

        prediction = model(input_image, training=False)
        detections = decoder(input_image, prediction)

        num_detections = detections.valid_detections[0]

        classes = {str(value): key for key, value in cls_ids.items()}
        class_names = [classes[str(int(x))] for x in detections.nmsed_classes[0][:num_detections].numpy()]

        input_image = input_image[0].numpy()
        input_image = (input_image * 255).astype(np.uint8)

        visualize_detections(
            input_image,
            detections.nmsed_boxes[0][:num_detections],
            class_names,
            detections.nmsed_scores[0][:num_detections],
        )


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        default='/home/david/Projects/RetinaNet-Tensorflow/models/2020-10-23_16-27-56',
        type=str,
        help='path to the directory where model was saved',
    )
    parser.add_argument(
        '--split',
        default='test',
        type=str,
        help='train or test',
    )
    parser.add_argument(
        '--record_dir',
        default='records',
        type=str,
        help='directory where the records are',
    )
    parser.add_argument(
        '--record_name',
        default='hippo',
        type=str,
        help='name of the record without _train.tfrecord or _test.tfrecord',
    )
    parser.add_argument(
        '--n',
        default=10,
        type=int,
        help='how many images will be displayed',
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parseargs()
    inference(
        model_dir              = args.model_dir,
        split                  = args.split.lower(),
        record_dir             = args.record_dir,
        record_name            = args.record_name,
        number_of_images_shown = args.n,
    )