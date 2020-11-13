import os
import tensorflow as tf
import tensorflow_datasets as tfds

from params import Params as p

model = p.model

record_reader = p.record_reader

train_dataset = record_reader.read_record('train')
val_dataset = record_reader.read_record('test')

# epochs = 3
#
# model.fit(
#     train_dataset.take(100),
#     validation_data=train_dataset.take(10),
#     epochs=epochs,
#     callbacks=p.callbacks_list,
#     verbose=1,
# )


# @tf.function
def train_step(input, labels):
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        prediction = model(input, training=True)
        clf_loss, box_loss = p.loss_fn(labels, prediction)
        loss = clf_loss + box_loss

    gradients = tape.gradient(loss, model.trainable_variables)

    clipped_gradients = []
    for grad in gradients:
        clipped_gradients.append(tf.clip_by_value(grad, -2, 2))

    p.optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))


# @tf.function
def test(dataset, metric1, metric2, sum_metric):

    for i, (image, label) in enumerate(dataset):
        if i >= 50: break
        prediction = model(image, training=False)
        clf_loss, box_loss = p.loss_fn(label, prediction)
        metric1(clf_loss)
        metric2(box_loss)
        sum_metric(clf_loss + box_loss)

file_writer = tf.summary.create_file_writer(p.model_dir + "/metrics")
file_writer.set_as_default()
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=p.model_dir)

train_clf_loss = tf.keras.metrics.Mean(name='train_clf_loss')
train_box_loss = tf.keras.metrics.Mean(name='train_box_loss')
test_clf_loss = tf.keras.metrics.Mean(name='test_clf_loss')
test_box_loss = tf.keras.metrics.Mean(name='test_box_loss')

train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss  = tf.keras.metrics.Mean(name='test_loss')

def foo():
    for i, (image, label) in enumerate(train_dataset):

        print ('i', i, end='\r')

        train_step(image, label)

        if i % 1000 == 0:
            test(val_dataset, test_clf_loss, test_box_loss, test_loss)
            test(train_dataset, train_clf_loss, train_box_loss, train_loss)

            tf.summary.scalar('test_clf_loss', test_clf_loss.result(), i)
            tf.summary.scalar('test_box_loss', test_box_loss.result(), i)
            tf.summary.scalar('test_loss', test_loss.result(), i)

            tf.summary.scalar('train_clf_loss', train_clf_loss.result(), i)
            tf.summary.scalar('train_box_loss', train_box_loss.result(), i)
            tf.summary.scalar('train_loss', train_loss.result(), i)

            test_clf_loss.reset_states()
            test_box_loss.reset_states()
            test_loss.reset_states()
            train_clf_loss.reset_states()
            train_box_loss.reset_states()
            train_loss.reset_states()

        if i % 5000 == 0:
            model.save_weights(os.path.join(p.model_dir, 'model_' + str(i)), save_format='tf')

            model.save(os.path.join(p.model_dir, 'model_' + str(i)))

foo()