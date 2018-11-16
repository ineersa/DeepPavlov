# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
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

from typing import Optional, Union

import tensorflow as tf
import tensorflow_hub as hub


class TFHubClassifier(tf.keras.Model):
    """
    A classifier that uses TensorFlow Hub module as the embedder.

    Args:
        embedder: TF Hub module for text embedding (default signature is used); currently could not be trained
        encoder: a :class:`~tf.keras.Layer` (or the whole :class:`~tf.keras.Model`) waiting for input with the shape
            corresponding to the embedder output shape
        num_classes: currently only softmax over this number of neurons in the head of the classifier is implemented
    """
    def __init__(self,
                 embedder: Union[str, hub.Module] = None,
                 encoder: Optional[tf.keras.Layer] = None,
                 num_classes: int = 2
                 ) -> None:
        super().__init__(name='tf_hub_classifier')
        self.embedder = hub.Module(embedder, trainable=False) if type(embedder) is str else embedder
        self.encoder = encoder or tf.keras.layers.Lambda(lambda x: x)
        self.num_classes = num_classes
        self.head = tf.keras.layers.Dense(self.num_classes, activation='softmax')

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor:
        emb = self.embedder(inputs)
        h = self.encoder(emb)
        return self.head(h)
