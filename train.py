import os
import tensorflow as tf
import tensorflow_datasets as tfds

from params import Params as p

model = p.model

record_reader = p.record_reader

train_dataset = record_reader.read_record('train')
val_dataset = record_reader.read_record('test')

epochs = 10

# model.fit(
#     train_dataset.take(100),
#     validation_data=train_dataset.take(10),
#     epochs=epochs,
#     callbacks=p.callbacks_list,
#     verbose=1,
# )

i = 0

def train_step(input, labels):
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        prediction = model(input, training=True)
        loss = p.loss_fn(labels, prediction)

    gradients = tape.gradient(loss, model.trainable_variables)

    clipped_gradients = []
    for grad in gradients:
        clipped_gradients.append(tf.clip_by_value(grad, -2, 2))

    p.optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))
    print (i, loss.numpy())

for image, label in train_dataset:
    train_step(image, label)
    i += 1

    if i % 1000 == 0:
        model.save_weights(os.path.join(p.model_dir, 'model_' + str(i)), save_format='tf')
