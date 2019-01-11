# Copyright 2019 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
from tensorflow.python.keras import backend as K

from deeppavlov.contrib.data.classification.rusentiment import train_input_fn  # , test_input_fn
from deeppavlov.contrib.models.classifiers import TFHubRawTextClassifier


# tf.enable_eager_execution()
if tf.executing_eagerly():
    raise RuntimeError('This example currently doesn\'t work in Eager mode.')

# We still need to manage the session explicitly (will definitely not be necessary in TF2.0), mainly because of lookup
# ops both in data pipeline and model (TF Hub)
sess = tf.Session()
K.set_session(sess)

# By TF conventions, calling input_fn produces tf.data.Dataset object, however input_fn is also should be suitable
# for tf.estimator.Estimator.train() method
train_data = train_input_fn()
# test_data = test_input_fn()

# Instantiation of a model class preferably should be possible without passing any arguments (every arguments should
# have reasonable default values); Keras inherits this design principle from scikit-learn. We need some non-defaults...
sentiment_analyzer = TFHubRawTextClassifier(
    tfhub_spec='http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-twitter_2013-01_2018-04_600k_steps.tar.gz',
    num_classes=5
)

# TF developers are going to unify in TF2.0 native TF losses, optimizers and metrics with corresponding Keras versions
sentiment_analyzer.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

# Train for just a couple of batches for demonstration purposes
sentiment_analyzer.fit(train_data, steps_per_epoch=2)

# This is similar to sentiment_analyzer.predict(), but works for now without additional reinitialization trick
print(sentiment_analyzer(['привет! Как дела?']).eval(session=sess))
