from __future__ import absolute_import, division

from visual_model_selector import ModelFactory
from glob import glob
from configs import argHandler #Import the default arguments
from model_utils import set_gpu_usage, get_optimizer, create_class_weight, get_generator
import numpy as np
import math
from augmenter import augmenter, augmenter_flip
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard, CSVLogger
import os
import json

FLAGS = argHandler()
FLAGS.setDefaults()

set_gpu_usage(FLAGS.gpu_percentage)

model_factory=ModelFactory()

try:
    os.makedirs(FLAGS.save_model_path)
except:
    print("path already exists")

train_generator = get_generator(FLAGS, FLAGS.train_csv, augmenter,augmenter_flip, shuffle_on_end=True)
test_generator = get_generator(FLAGS, FLAGS.test_csv)

class_weights = None
if FLAGS.class_weights_balancing:
    class_counts = train_generator.get_class_counts()
    class_weights = create_class_weight(class_counts,1.0)
    
#load classifier from saved weights or get a new one
if FLAGS.load_model_path != '' and FLAGS.load_model_path is not None:
    visual_model = load_model(FLAGS.load_model_path)
    if FLAGS.show_model_summary:
        visual_model.summary()
else:
    visual_model = model_factory.get_model(FLAGS)
    
opt = get_optimizer(FLAGS.optimizer_type, FLAGS.learning_rate)
    
visual_model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
checkpoint = ModelCheckpoint(os.path.join(FLAGS.save_model_path, 'best_model.hdf5'),monitor='val_acc',save_best_only=True, save_weights_only=False, mode='max',verbose=1)

with open(os.path.join(FLAGS.save_model_path,'configs.json'), 'w') as fp:
    json.dump(FLAGS, fp, indent=4)

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=FLAGS.learning_rate_decay_factor, patience=FLAGS.reduce_lr_patience,
                      verbose=1, mode="min", min_lr=FLAGS.minimum_learning_rate),
    checkpoint,
    CSVLogger(os.path.join(FLAGS.save_model_path,'training_log.csv')),
    TensorBoard(log_dir=os.path.join(FLAGS.save_model_path, "logs"), batch_size=FLAGS.batch_size)
]

visual_model.fit_generator(
    generator=train_generator,
    steps_per_epoch=train_generator.steps,
    epochs=FLAGS.num_epochs,
    validation_data=test_generator,
    validation_steps=test_generator.steps,
    workers=FLAGS.generator_workers,
    callbacks=callbacks,
    max_queue_size=FLAGS.generator_queue_length,
    class_weight=class_weights,
    shuffle=False
)

            
 