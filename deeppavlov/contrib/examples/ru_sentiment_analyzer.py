# Copyright 2018 Neural Networks and Deep Learning lab, MIPT
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
from tensorflow.keras import backend as K

from deeppavlov.contrib.data.classification.rusentiment import train_input_fn as get_train_data
from deeppavlov.contrib.models.classifiers import TFHubRawTextClassifier


# tf.enable_eager_execution()
if tf.executing_eagerly():
    raise RuntimeError('This example currently doesn\'t work in Eager mode.')

sess = tf.Session()
K.set_session(sess)

train_data = get_train_data()

sentiment_analyzer = TFHubRawTextClassifier(
    tfhub_spec='http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-twitter_2013-01_2018-04_600k_steps.tar.gz',
    num_classes=5
)

sentiment_analyzer.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

sentiment_analyzer.fit(train_data, steps_per_epoch=2)
