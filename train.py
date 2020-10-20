import tensorflow as tf
import tensorflow_datasets as tfds

from params import Params as p

model = p.model

record_reader = p.record_reader

train_dataset = record_reader.read_record('train')
val_dataset = record_reader.read_record('test')

epochs = 3

model.fit(
    train_dataset.take(1000),
    validation_data=train_dataset.take(1),
    epochs=epochs,
    callbacks=p.callbacks_list,
    verbose=1,
)
