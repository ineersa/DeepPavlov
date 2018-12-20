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

from typing import Optional, Union, Mapping, Collection

import tensorflow as tf
import tensorflow_hub as hub


class TFHubRawTextEmbedder(tf.keras.layers.Layer):
    """
    This layer wraps TF Hub text embedding module.

    Args:
        tfhub_spec: TF Hub eligible value (url, filepath, ModuleSpec object)
        trainable: boolean flag responsible for the training of TF Hub module weights
        tags: a collection of strings corresponding to graph variants of the module; passed to the module initialization
        signature: a signature of TF Hub module to retrieve output dict from
        out_info_key: a key in the output
        kwargs: keyword arguments passed to :meth:`super().__init__`

    Todo:
        elaborate on different signatures, output keys and tags of TF Hub modules
    """
    def __init__(self,
                 tfhub_spec: Union[str, hub.ModuleSpec],
                 trainable: bool = False,
                 name: str = 'tfhub_text_module',
                 tags: Optional[Collection[str]] = None,
                 signature: str = 'default',
                 out_info_key: str = 'default',
                 **kwargs) -> None:

        self.hub_module = hub.Module(tfhub_spec, trainable=trainable, name=name, tags=tags)

        if signature == 'default' and out_info_key == 'default':
            self.signature = signature
            self.out_info_key = out_info_key
        else:
            raise NotImplementedError('Currently, only the default signature of TF Hub module '
                                      'with the default output key is supported')

        self._out_info = self.hub_module.get_output_info_dict()[signature]

        super(TFHubRawTextEmbedder, self).__init__(**kwargs)

    def call(self, inputs: Union[tf.Tensor, Mapping[str, tf.Tensor]], mask=None) -> tf.Tensor:
        """
        Signatures with multiple inputs wait for input as a dictionary, however the default signature of all TF hub
        modules (as far as I know) waits for usual tensor. Currently, output is also tensor, but, probably, it could be
        useful for some cases to obtain output as a dictionary also.
        """
        return self.hub_module(inputs,
                               as_dict=True,
                               signature=self.signature,
                               )[self.out_info_key]

    def compute_output_shape(self, input_shape):
        return self._out_info.get_shape()
