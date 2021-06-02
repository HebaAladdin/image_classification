from __future__ import absolute_import, division

from visual_model_selector import ModelFactory
from configs import argHandler  # Import the default arguments
from model_utils import set_gpu_usage, get_evaluation_metrics, get_generator, get_optimizer
from tensorflow.keras.models import load_model
from tensorflow.keras import metrics


FLAGS = argHandler()
FLAGS.setDefaults()

set_gpu_usage(FLAGS.gpu_percentage)

model_factory = ModelFactory()



train_generator = get_generator(FLAGS, FLAGS.train_csv)
test_generator = get_generator(FLAGS, FLAGS.test_csv)

if FLAGS.load_model_path != '' and FLAGS.load_model_path is not None:
    visual_model = load_model(FLAGS.load_model_path)
    if FLAGS.show_model_summary:
        visual_model.summary()
else:
    visual_model = model_factory.get_model(FLAGS)

opt = get_optimizer(FLAGS.optimizer_type, FLAGS.learning_rate)

visual_model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
def get_metrics_from_generator(generator, verbose=1):
    y_hat = visual_model.predict_generator(generator, steps=generator.steps, workers=FLAGS.generator_workers,
                                           max_queue_size=FLAGS.generator_queue_length, verbose=verbose)
    y_hat = y_hat.argmax(axis=1)
    y = generator.get_y_true()
    get_evaluation_metrics(y_hat, y, FLAGS.classes)


print("***************Train Metrics*********************")
get_metrics_from_generator(train_generator)
print("***************Test Metrics**********************")
get_metrics_from_generator(test_generator)