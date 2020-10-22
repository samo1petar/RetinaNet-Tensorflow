import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from lib.layers.DecodePredictions import DecodePredictions
from lib.image import resize_and_pad_image
from lib.visualize import visualize_detections
from lib.data.classes import classes
from lib.data.hippo_classes import cls_ids

from params import Params as p


# Change this to `model_dir` when not using the downloaded weights
weights_dir = p.model_dir

latest_checkpoint = tf.train.latest_checkpoint(weights_dir)

model = p.model
model.load_weights(latest_checkpoint)

# image = tf.keras.Input(shape=[None, None, 3], name="image")
# predictions = model(image, training=False)
# detections = DecodePredictions(confidence_threshold=0.5)(image, predictions)
# inference_model = tf.keras.Model(inputs=image, outputs=detections)

decoder = DecodePredictions(confidence_threshold=0.5)

def prepare_image(image):
    image, _, ratio = resize_and_pad_image(image, jitter=None)
    image = tf.keras.applications.resnet.preprocess_input(image)
    return tf.expand_dims(image, axis=0), ratio

train_dataset = p.record_reader.read_record('train')

i = 0

for input_image, _ in train_dataset:
    if i >= 10: break
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
