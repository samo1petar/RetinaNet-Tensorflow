import os
import tensorflow as tf
import tensorflow_datasets as tfds

# setup limited gpu usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

from params import Params as p
from shutil import copyfile

# if model_dir dosen't exists (and it dosen't because it's unique every time) -> create it
if not os.path.exists(p.model_dir):
    os.mkdir(p.model_dir)
# save parameters to model_dir so parameters don't get lost
copyfile('params.py', os.path.join(p.model_dir, 'params.py'))
#
model = p.model
#
record_reader = p.record_reader
# load_train and test datasets (test is also called validation)
train_dataset = record_reader.read_record('train')
test_dataset = record_reader.read_record('test')
# create tensorboard metrics file
file_writer = tf.summary.create_file_writer(p.model_dir + "/metrics")
file_writer.set_as_default()
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=p.model_dir)
# setup tensorboard mean metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss  = tf.keras.metrics.Mean(name='test_loss')

'''
Train steps:
1. Forward pass. Run input through the network and get the output
2. Compare output and targets (targets are desired output) and get their differences (this is called loss)
3. Run backpropagation algorithm to get gradients on all parameters from the network
(3.1 optional -> clip the gradients, this is useful if outliers occur in the data)
4. Update the parameters
'''
@tf.function
def train_step(input, labels):
    # GradientTape tells to the tensorflow to remember the gradients during forward pass
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        # 1. forward pass -> outputs prediction
        prediction = model(input, training=True)
        # 2. calculate loss
        loss = p.loss_fn(labels, prediction)
    # 3. get gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    # 3.1 clip gradients
    clipped_gradients = []
    for grad in gradients:
        clipped_gradients.append(tf.clip_by_value(grad, -2, 2))
    # 4. update parameters (aka variables)
    p.optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))

'''
Test steps:
1. Forward pass. Run input through the network and get the output
2. Compare output and targets (targets are desired output) and get their differences (this is called loss)
3. Save loss
'''
@tf.function
def test_step(dataset, metric):

    for i, (image, label) in enumerate(dataset):
        if i >= 50: break
        # 1. forward pass
        prediction = model(image, training=False)
        # 2. calculate loss
        loss = p.loss_fn(label, prediction)
        # 3. save loss
        metric(loss)

def train():
    # iterate over train dataset
    for i, (image, label) in enumerate(train_dataset):
        # print iteration number
        print ('i', i, end='\r')
        # train step
        train_step(image, label)
        # every 1000 iteration evaluate test and train
        if i % 1000 == 0:
            test_step(test_dataset, test_loss)
            test_step(train_dataset, train_loss)
            # save average test and train loss
            tf.summary.scalar('test_loss', test_loss.result(), i)
            tf.summary.scalar('train_loss', train_loss.result(), i)
            # reset metrics
            test_loss.reset_states()
            train_loss.reset_states()
        # every 5000 iteration save model weights in two formats
        if i % 5000 == 0:
            model.save_weights(os.path.join(p.model_dir, 'model_' + str(i)), save_format='tf')
            model.save(os.path.join(p.model_dir, 'model_' + str(i)))

if __name__ == '__main__':
    train()