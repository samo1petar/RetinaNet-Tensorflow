import tensorflow as tf
from lib.visualize import visualize_detections
import numpy as np
import cv2
from lib.data.hippo_classes import cls_ids
from lib.LabelEncoder import LabelEncoder
from IPython import embed
from lib.data.RecordReader import RecordReader

def show(image):
    cv2.imshow('', image)
    cv2.waitKey()
    cv2.destroyAllWindows()


record_reader = RecordReader(
        record_dir           = 'records',
        record_name          = 'hippo',
        label_encoder        = LabelEncoder(),
        batch_size           = 1,
        shuffle_buffer       = 1,
        num_parallel_calls   = 1,
        num_parallel_reads   = 1,
        prefatch_buffer_size = 1,
    )

train_dataset = record_reader.read_record('test')

i = 0
for image, bbox, cls in train_dataset:
    if i >= 50: break
    i += 1

    print (image.shape)

    image = (image[0].numpy() * 255).astype(np.uint8)
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

'''
[[191.48436 , 206.31874 , 381.61874 , 178.025   ],
 [294.29376 ,  73.84376 , 198.925   , 142.3     ],
 [254.51251 , 219.56876 , 197.02502 , 152.55    ],
 [135.46251 , 125.356255, 270.92502 , 233.825   ],
 [258.42188 ,  39.453125,  46.59375 ,  28.456251],
 [308.94687 ,  38.628128,  35.668762,  28.54375 ],
 [267.00626 ,  67.75938 ,  51.887512,  43.443745],
 [256.83124 ,  23.453125,  58.600006,  43.793747]]
 
[[  0.675   , 117.30625 , 382.29373 , 295.33124 ],
 [194.83125 ,   2.69375 , 393.75626 , 144.99376 ],
 [156.      , 143.29375 , 353.02502 , 295.84375 ],
 [  0.      ,   8.44375 , 270.92502 , 242.26875 ],
 [235.125   ,  25.225   , 281.71875 ,  53.68125 ],
 [291.1125  ,  24.35625 , 326.78125 ,  52.9     ],
 [241.0625  ,  46.037502, 292.95    ,  89.48125 ],
 [227.53125 ,   1.55625 , 286.13126 ,  45.35    ]]

['bowl', 'bowl', 'broccoli', 'bowl', 'orange', 'orange', 'orange', 'orange']
[45, 45, 50, 45, 49, 49, 49, 49]

[  orange   [ 2.2760611e+02,  1.5918655e+00,  2.8609482e+02,  4.5373684e+01], +
   orange   [ 2.3515399e+02,  2.5172421e+01,  2.8169550e+02,  5.3697506e+01], +
   orange   [ 2.4108517e+02,  4.6018394e+01,  2.9294748e+02,  8.9513344e+01], +
   bowl     [ 1.9496555e+02,  2.5139771e+00,  3.9389035e+02,  1.4522272e+02], +
   bowl     [-3.3058167e-01,  8.9468307e+00,  2.7135480e+02,  2.4219626e+02], +
   bowl     [ 7.1324158e-01,  1.1752789e+02,  3.8145520e+02,  2.9511127e+02], +
   broccoli [ 1.5580937e+02,  1.4302332e+02,  3.5292450e+02,  2.9607584e+02], +
   orange   [ 2.9110397e+02,  2.4325800e+01,  3.2676175e+02,  5.2897385e+01], +
   bowl     [ 5.7750885e+01,  6.5714577e+01,  2.9877313e+02,  2.7723166e+02],
   orange   [ 2.3769647e+02,  3.7375381e+01,  2.8628088e+02,  7.2752052e+01]] +/+

['orange', 'orange', 'orange', 'bowl', 'bowl', 'bowl', 'broccoli', 'orange', 'bowl', 'orange']
[49., 49., 49., 45., 45., 45., 50., 49., 45., 49.]

'''