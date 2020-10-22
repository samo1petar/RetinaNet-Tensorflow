import os
import tensorflow as tf

from lib.data.RecordReader import RecordReader
from lib.data.RecordWriter import RecordWriter
from lib.data.RecordWriterHippo import RecordWriterHippo
from lib.LabelEncoder import LabelEncoder
from lib.feature_extractor.backbone import get_backbone
from lib.loss.RetinaNetLoss import RetinaNetLoss
from lib.model.RetinaNet import RetinaNet


class Params:
    dataset = '/media/david/A/Datasets/COCO'

    records = 'records'
    results = 'results'

    record_name = 'hippo_ten'

    model_dir = 'models_hippo_ten' # TODO used for callback, should be dependent on each train

    label_encoder = LabelEncoder()

    num_classes = 12
    batch_size = 3

    record_writer = RecordWriterHippo(
        data_path           = dataset,
        record_dir          = records,
        record_name         = record_name,
        annotations         = '/media/david/A/Datasets/PlayHippo/detections.json',
        save_n_test_images  = 1,
        save_n_train_images = 10,
    )
    record_reader = RecordReader(
        record_dir           = records,
        record_name          = record_name,
        label_encoder        = label_encoder,
        batch_size           = batch_size,
        shuffle_buffer       = 1,
        num_parallel_calls   = 1,
        num_parallel_reads   = 1,
        prefatch_buffer_size = 1,
    )

    # learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
    learning_rates = [0.0025, 0.00125, 0.000625, 0.0003125]
    learning_rate_boundaries = [125, 250, 2000]
    learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=learning_rate_boundaries, values=learning_rates
    )

    resnet50_backbone = get_backbone()
    loss_fn = RetinaNetLoss(num_classes)
    model = RetinaNet(num_classes, resnet50_backbone)

    optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
    model.compile(loss=loss_fn, optimizer=optimizer)

    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
            monitor="loss",
            save_best_only=False,
            save_weights_only=True,
            verbose=1,
        )
    ]