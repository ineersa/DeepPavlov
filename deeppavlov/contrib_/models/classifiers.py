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

from typing import Dict, Optional, Union, List

import tensorflow as tf
import tensorflow_hub as hub

from deeppavlov.contrib.blocks.layers import TFHubRawTextEmbedder


class TFHubRawTextClassifier(tf.keras.Model):
    """
    Multilayer feed-forward network on top of :class:`deeppavlov.contrib.blocks.layers.TFHubRawTextEmbedder`

    Args:
        tfhub_spec: TF Hub eligible spec, if the text is not embedded inside data preparation pipeline
        train_emb: whether to update embedder weights jointly with classification head weights
        inp_key: the key to look for in the input dictionary; if not specified, the model waits for tensor as input
        hidden_dim: a size of the only hidden layer or list of sizes for multilayer encoder
        num_classes: number of logits returned by the model
    """
    def __init__(self,
                 tfhub_spec: Optional[Union[str, hub.Module]] = None,
                 train_emb: bool = False,
                 inp_key: Optional[str] = None,
                 hidden_dim: Union[int, List[int]] = 256,
                 num_classes: int = 2
                 ) -> None:
        super().__init__(name='tfhub_raw_text_classifier')

        # embed text if tfhub_spec is provided, or just pass through otherwise
        self.embedder = TFHubRawTextEmbedder(tfhub_spec=tfhub_spec,
                                             trainable=train_emb) if tfhub_spec else tf.keras.layers.Lambda(lambda x: x)
        self.inp_key = inp_key
        self.encoder = tf.keras.layers.Dense(units=hidden_dim, activation='relu')

        # we encourage to use losses that wait for logits
        self.head = tf.keras.layers.Dense(num_classes, activation=None)

    def call(self,
             inputs: Union[tf.Tensor, Dict[str, tf.Tensor]],
             training: bool = True,
             mask: Optional[Union[tf.Tensor, list]] = None
             ) -> tf.Tensor:
        if self.inp_key:
            inp_text = inputs[self.inp_key]
        else:
            inp_text = inputs
        emb = self.embedder(inp_text)
        h = self.encoder(emb)
        return self.head(h)
