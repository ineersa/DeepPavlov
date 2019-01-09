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

from deeppavlov.contrib.vocabulary import Vocabulary
from deeppavlov.contrib.data.sequence_tagging.conll2003 import train_input_fn
from deeppavlov.contrib.models.sequence_taggers import TFHubCRFSequenceTagger


# tf.enable_eager_execution()
if tf.executing_eagerly():
    raise RuntimeError('This example currently doesn\'t work in Eager mode.')

# We still need to manage the session explicitly (will definitely not be necessary in TF2.0), mainly because of lookup
# ops both in data pipeline and model (TF Hub)
sess = tf.Session()
K.set_session(sess)

# By TF conventions, calling input_fn produces tf.data.Dataset object, however input_fn is also suitable
# for tf.estimator.Estimator.train() method
train_data = train_input_fn()

# for now create label vocab by hand, however it will be
tag_vocab = Vocabulary(tf.constant(['O', 'B-LOC', 'B-PER', 'B-ORG', 'I-PER', 'I-ORG', 'B-MISC', 'I-LOC', 'I-MISC']))

# Instantiation of a model class preferably should be possible without passing any arguments (every arguments should
# have reasonable default values); Keras inherits this design principle from scikit-learn. We need some non-defaults...
ne_recognizer = TFHubCRFSequenceTagger(tag_vocab=tag_vocab, tfhub_spec='https://tfhub.dev/google/elmo/2')

# TF developers are going to unify in TF2.0 native TF losses, optimizers and metrics with corresponding Keras versions
ne_recognizer.compile(loss={'logits': ne_recognizer.crf.loss}, optimizer='adam', metrics=['accuracy'])

sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

# Train for just a couple of batches for demonstration purposes
ne_recognizer.fit(train_data, steps_per_epoch=2)

# This is similar to sentiment_analyzer.predict(), but works for now without additional reinitialization trick
print(ne_recognizer([['привет!', 'Как дела?']]).eval(session=sess))
