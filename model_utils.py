from __future__ import absolute_import, division

from skimage.transform import resize
from tensorflow.keras.models import model_from_json
import importlib
import os
import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import math
from generator import AugmentedImageSequence

def set_gpu_usage(gpu_memory_fraction):
    if gpu_memory_fraction <= 1 and gpu_memory_fraction > 0:
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
        sess = tf.Session(config=config)
    elif gpu_memory_fraction == 0:
        sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
    K.set_session(sess)
    
    
def get_optimizer(optimizer_type, learning_rate, lr_decay=0):
    optimizer_class = getattr(importlib.import_module("tensorflow.keras.optimizers"), optimizer_type)
    try:
        optimizer = optimizer_class(lr=learning_rate, decay=lr_decay)
    except:
        optimizer = optimizer_class(lr=learning_rate)
    return optimizer    



def classify_image(img, model,target_size=(224,224,3),classes=['left_foot','right_foot']):
    # resize
    img = img / 255.
    img = resize(img, target_size)
    batch_x = np.expand_dims(img, axis=0)
    #normalize
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    batch_x = (batch_x - imagenet_mean) / imagenet_std
    # predict
    predictions = model.predict(batch_x)
    predictions = np.argmax(predictions,axis=1)
    return np.array(classes)[predictions]

# predict on data from generator and calculate accuracy
def get_evaluation_metrics(predictions, labels, class_names):
    print(classification_report(labels, predictions, target_names=class_names))
    print("*******Confusion matrix*********")
    print(confusion_matrix(labels, predictions))
    print("\nAccuracy: %.2f" % accuracy_score(labels, predictions))


def get_generator(FLAGS, csv_path, data_augmenter=None,augmenter_flip=None,shuffle_on_end=False):
    return AugmentedImageSequence(
        dataset_csv_file=csv_path,
        label_columns=FLAGS.csv_label_columns,
        class_names=FLAGS.classes,
        flipped_class_names=FLAGS.flipped_classes,
        source_image_dir=FLAGS.image_directory,
        batch_size=FLAGS.batch_size,
        target_size=FLAGS.image_target_size,
        augmenter=data_augmenter,
        flip_augmenter=augmenter_flip,
        shuffle_on_epoch_end=shuffle_on_end,
    )

def create_class_weight(labels_count,mu=0.15):
    total = np.sum(labels_count)
    class_weight = dict()

    for key in range(len(labels_count)):
        score = math.log(mu*total/float(labels_count[key]))
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight