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

<<<<<<< HEAD
from logging import getLogger
=======
>>>>>>> origin/feature/contrib
from typing import Mapping, Optional, Union

import tensorflow as tf
import tensorflow_hub as hub

from deeppavlov.contrib.vocabulary import Vocabulary
from deeppavlov.contrib.blocks.layers import ELMoSequenceEmbedder
# from deeppavlov.contrib.blocks.crf import CRF
from deeppavlov.contrib.blocks.crf_layer import CRFLayer


<<<<<<< HEAD
logger = getLogger(__name__)


=======
>>>>>>> origin/feature/contrib
class TFHubCRFSequenceTagger(tf.keras.Model):
    """
    This tagger uses Conditional Random Fields on top of (optionally) encoded sequence of embeddings obtained with
    :class:`deeppavlov.contrib.blocks.layers.ELMoSequenceEmbedder`

    Args:
        tag_vocab: vocabulary that makes mapping between string representations of labels and their indexes
            (in both directions)
        tokens_inp_key: the key to look for in the input dictionary; if not specified, the model use inputs as tensors
        tfhub_spec: TF Hub eligible spec, if the text is not embedded inside data preparation pipeline
        train_emb: whether to update embedder weights jointly with classification head weights
        encoder: if not specified, CRF works on embeddings as features, however you can put here your BiLSTM,
            Transformer or something else
    """
    def __init__(self,
                 tag_vocab: Vocabulary,
                 tokens_inp_key: str = 'tokens',
                 tfhub_spec: Optional[Union[str, hub.Module]] = None,
                 train_emb: bool = False,
                 encoder: Optional[tf.keras.layers.Layer] = None
                 ) -> None:
        super().__init__(name='tfhub_sequence_tagger')

        self.tag_vocab = tag_vocab

        self.tokens_inp_key = tokens_inp_key
        # embed text if tfhub_spec is provided, or just pass through otherwise
        self.embedder = ELMoSequenceEmbedder(tfhub_spec=tfhub_spec,
                                             trainable=train_emb) if tfhub_spec else tf.keras.layers.Lambda(lambda x: x)

        self.encoder = encoder or tf.keras.layers.Lambda(lambda x: x)

<<<<<<< HEAD
        self.tag_projection_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=9))

        # self.crf = CRF(units=tag_vocab.size())
        self.crf = CRFLayer(name='crf')
=======
        # self.crf = CRF(units=tag_vocab.size())
        self.crf = CRFLayer()
>>>>>>> origin/feature/contrib

    def call(self,
             inputs: Union[tf.Tensor, Mapping[str, tf.Tensor]],
             training: bool = True,
             mask: Optional[Union[tf.Tensor, list]] = None
             ) -> Mapping[str, tf.Tensor]:
<<<<<<< HEAD
        # if self.tokens_inp_key:
        #     inp_text = inputs[self.tokens_inp_key]
        # else:
        #     inp_text = inputs
        emb = self.embedder(inputs)
        h = self.encoder(emb)
        p = self.tag_projection_layer(h)
        humanized_tags = self.tag_vocab.reverse_lookup(tf.constant([1, 2, 3], dtype=tf.int64))
        return {"logits": self.crf([p, inputs['sequence_len']]), "tags": humanized_tags}
=======
        if self.tokens_inp_key:
            inp_text = inputs[self.tokens_inp_key]
        else:
            inp_text = inputs
        emb = self.embedder(inp_text)
        h = self.encoder(emb)
        humanized_tags = self.tag_vocab.reverse_lookup(tf.constant([1, 2, 3]))
        return {'logits': self.crf(h), 'tags': humanized_tags}
>>>>>>> origin/feature/contrib
