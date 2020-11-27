import argparse
import numpy as np

from lib.bbox import convert_to_corners
from lib.data.hippo_classes import cls_ids
from lib.data.RecordReader import RecordReader
from lib.LabelEncoder import LabelEncoder
from lib.visualize import visualize_detections


'''
This script is for visualizing records for better understanding what the network sees.
'''


def test_record(
        split: str,
        record_dir: str,
        record_name: str,
        number_of_images_shown: int,
):
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

    train_dataset = record_reader.read_record(split, create_labels=False)

    i = 0
    for image, bbox, cls in train_dataset:
        if i >= number_of_images_shown: break
        i += 1

        image = (image[0].numpy() * 255).astype(np.uint8)

        bbox = convert_to_corners(bbox)

        bbox = bbox.numpy()
        cls = cls.numpy()

        classes = {str(value): key for key, value in cls_ids.items()}
        class_names = [classes[str(int(x))] for x in cls[0]]

        visualize_detections(
                image,
                bbox[0],
                class_names,
                [1] * len(class_names),
            )


def parseargs():
    parser = argparse.ArgumentParser()
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
    test_record(
        split                  = args.split.lower(),
        record_dir             = args.record_dir,
        record_name            = args.record_name,
        number_of_images_shown = args.n,
    )