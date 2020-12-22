import os
import tensorflow as tf

from lib.data.RecordReader import RecordReader
from lib.data.RecordWriter import RecordWriter
from lib.LabelEncoder import LabelEncoder
from lib.feature_extractor.backbone import get_backbone, get_backbone_MobileNet_v2, \
    get_backbone_conv_small, get_backbone_conv_basic
from lib.loss.RetinaNetLoss import RetinaNetLoss
from lib.model.RetinaNet import RetinaNet
from lib.tools.time import get_time


class Params:

    dataset = '/home/david/Projects/Retinus/dataset/images'
    annotations = '/home/david/Projects/Retinus/dataset/annotations_all'

    records = '/home/david/Projects/Retinus/dataset/records'
    record_name = 'retinus_v2'

    model_dir = '/home/david/Projects/Retinus/dataset/models/' + get_time() # TODO makedirs if not exists

    label_encoder = LabelEncoder()

    num_classes = 19
    batch_size = 4

    record_writer = RecordWriter(
        dataset             = dataset,
        annotations         = annotations,
        record_dir          = records,
        record_name         = record_name,
        save_n_test_images  = None,
        save_n_train_images = None,
    )
    # RecordReader is a class that knows how to read records. Once the records are created, only the RecordReader is
    #   used, not RecordWriterHippo
    record_reader = RecordReader(
        record_dir           = records,
        record_name          = record_name,
        label_encoder        = label_encoder,
        batch_size           = batch_size,
        shuffle_buffer       = 20,
        num_parallel_calls   = 4,
        num_parallel_reads   = 4,
        prefetch_buffer_size = 20,
    )

    # learning rate rate, used for experiments
    lr_rate = 1
    # learning rates, after each step in `learning_rate_boundaries` the next learning rate is used. Usually learning
    #   rate is being multiplied with batch_size, because that way the direction is more precise
    learning_rates = [0.0025 * batch_size * lr_rate, 0.00125 * batch_size * lr_rate, 0.000625 * batch_size * lr_rate, 0.0003125 * batch_size * lr_rate]
    # steps when the next learning rate is being used
    learning_rate_boundaries = [125, 250, 2000]
    learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=learning_rate_boundaries, values=learning_rates
    )
    # define model backbone. Most of the model weights are here.
    backbone = get_backbone_MobileNet_v2()
    # define loss function
    loss_fn = RetinaNetLoss(num_classes)
    # create the model
    model = RetinaNet(num_classes, backbone, feature_pyramid_channels=32, head_channels=32, head_depth=4)
    # define optimizer. In this case Gradient descent with momentum is used.
    optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
    # compile the model. Here, everything gets glue up.
    model.compile(loss=loss_fn, optimizer=optimizer)
    # callbacks are only used if training is called with .fit() method. And here it's not as default.
    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
            monitor="loss",
            save_best_only=False,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=model_dir),
    ]
