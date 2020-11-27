# RetinNet-Tensorflow

This repo is created with [keras RetinNet tutorial](https://keras.io/examples/vision/retinanet/).

## Install

Install with virtualenv.

```
$ git clone https://github.com/samo1petar/RetinaNet-Tensorflow.git
$ cd RetinaNet-Tensorflow
$ mkdir venv
$ python3 -m virtualenv venv
$ source venv/bin/activate
$ pip -r install requirements.txt
```

## Setup and Easy run

After install, create directories for records and models, and download annotations.

```
$ cd .../RetinaNet-Tensorflow
$ mkdir records
$ mkdir models
```

Download model from [here](https://drive.google.com/drive/folders/1grErzQs_62OiPLjWKoMU8mjBoY5IT0CV?usp=sharing)
and place it inside `models` directory.

Download splits from [here](https://drive.google.com/drive/folders/1vWjlLVoyyX1jjuL4E8dr8JYI24WJ4dyt?usp=sharing)
and place them inside `records` directory.

Download `detections.json` from [here](https://drive.google.com/file/d/1LnbzcsvulPii5bLFbzFpIinkkn5YDIub/view?usp=sharing)
and place it inside project directory.

#### Inference

To visualize how well the model runs:

```
$ python infenrece.py --model_dir /path/to/repo/models/2020-10-23_16-27-56
```
Replace `/path/to/repo` with actual repository path.

#### Train

Change `annotations` parameter to where you put `detections.json`. If you followed the tutorial, it should
be `/path/to/repo/detections.json`. Also, if you are reusing the records, it will not use `detections.json`
but it needs to be there.

```
$ python train.py
```


## Train

Train script doesn't have command like parameters. There are too many of them.
Instead, all parameters are grouped in params.py file.

Here are most important:
- records - name of the directory where records will be stored (recommended is /.../RetinaNet-Tensorflow/records)
- record_name - name of the record. If the record already exists, it will reuse it
- model_dir - directory where train experiment will be saved. It is best to leave it as `models/` + get_time() 
and have everything in the save repo
- num_classes - number of output classes. For animals it's 12
- batch_size - batch size (default: 4)
- annotations - json file with all the data information. See 
[link](https://drive.google.com/file/d/1LnbzcsvulPii5bLFbzFpIinkkn5YDIub/view?usp=sharing) for reference.
- backbone - the model backbone is set up here. It depends on the problem and the amount of data. In this case
I used default, which is `get_backbone()` function. But there are smaller ones, such as `get_backbone_MobileNet_v2()`, 
`get_backbone_conv_small()` and `get_backbone_conv_basic()`
- feature_pyramid_channels - this is a parameter that defines the channels of the convolutions of upward part 
of the model. Basically what it means is, the bigger the number the bigger the model will be. In should be balanced
with models backbone. 256 is a too much for smaller backbones. So if you are going to try other backbones, lower this
to [32 - 128] range.
- head_channels - similar as feature_pyramid_channels. This is the size of the channels in the HEAD of the model.
The model HEAD come after the feature_pyramid, so it describes the size of the network after the pyramid. The model
has two HEADS, one for classification and one for regression.
- head_depth - if head channels describes the HEADS width, head_depth is HEADS length

```
[set up parameters in params.py]
$ python train.py
```

## Inference

Inference script is for visualizing how well model works. 
It runs the model and draws the detections over the image.

```
usage:
$ python inference.py [-h] [--model_dir MODEL_DIR] [--split SPLIT]
                    [--record_dir RECORD_DIR] [--record_name RECORD_NAME]
                    [--n N]

arguments:
  -h, --help    show this help message and exit
  --model_dir   path to the directory where model was saved
  --split       train or test (default: test)
  --record_dir  directory where the records are (default: records)
  --record_name name of the record without '_train.tfrecord' or '_test.tfrecord' (default: hippo)
  --n           how many images will be displayed (default: 10)

```

## Test record

Test record script is for visualizing records for better understanding what the network sees.

```
usage:
$ python test_record.py [-h] [--split SPLIT] [--record_dir RECORD_DIR]
               [--record_name RECORD_NAME] [--n N]

arguments:
  -h, --help    show this help message and exit
  --split       train or test (default: test)
  --record_dir  directory where the records are (default: records)
  --record_name name of the record without '_train.tfrecord' or '_test.tfrecord' (default: hippo)
  --n           how many images will be displayed (default: 10)

```

## Conversion to TFLite

The optimized TFLite model is [here](https://drive.google.com/file/d/1bEIwmY7nONMYtp-JiO1nfdyeuUKGsu2G/view?usp=sharing).
Use script bellow to convert the trained model to tensorflow lite.

Note. Use tensorflow version 2.3.0 for conversion.

```
import tensorflow as tf

saved_model_dir = '/path/to/repo/models/year-mm-dd_hh-min-sec/model_{iteration}'
output_model_file = 'converted_model.tflite'

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.allow_custom_ops = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
open(output_model_file, "wb").write(tflite_model)
```
    





