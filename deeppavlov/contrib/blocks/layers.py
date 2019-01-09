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

from typing import Optional, Union, Mapping, Collection, Any

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

    def call(self,
             inputs: Union[tf.Tensor, Mapping[str, tf.Tensor]],
             mask: Optional[Union[tf.Tensor, list]] = None
             ) -> tf.Tensor:
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


class ELMoSequenceEmbedder(tf.keras.layers.Layer):
    """
    This layer wraps TF Hub ELMo embedding module.

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
                 signature: str = 'tokens',
                 out_info_key: str = 'elmo',
                 **kwargs) -> None:

        self.hub_module = hub.Module(tfhub_spec, trainable=trainable, name=name, tags=tags)

        if signature == 'tokens' and out_info_key == 'elmo':
            self.signature = signature
            self.out_info_key = out_info_key
        else:
            raise NotImplementedError('Currently, only the "tokens" signature of TF Hub module '
                                      'with the "elmo" output key is supported')

        self._out_info = self.hub_module.get_output_info_dict()[out_info_key]

        super(ELMoSequenceEmbedder, self).__init__(**kwargs)

    def call(self,
             inputs: Mapping[str, tf.Tensor],
             mask: Optional[Union[tf.Tensor, list]] = None
             ) -> tf.Tensor:
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


class LayerNormalization(tf.keras.layers.Layer):

    def __init__(self,
                 center: bool = True,
                 scale: bool = True,
                 epsilon: Optional[float] = None,
                 gamma_initializer: str = 'ones',
                 beta_initializer: str = 'zeros',
                 gamma_regularizer: Optional[str] = None,
                 beta_regularizer: Optional[str] = None,
                 gamma_constraint: Optional[str] = None,
                 beta_constraint: Optional[str] = None,
                 **kwargs) -> None:
        """
        Layer normalization layer
        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)

        This implementation is heavily borrowed from https://github.com/CyberZHG/keras-layer-normalization

        Args:
            center: If True, add offset of `beta` to normalized tensor. If False, `beta` is ignored.
            scale: If True, multiply by `gamma`. If False, `gamma` is not used.
            epsilon: Epsilon for variance.
            gamma_initializer: Initializer for the gamma weight.
            beta_initializer: Initializer for the beta weight.
            gamma_regularizer: Optional regularizer for the gamma weight.
            beta_regularizer: Optional regularizer for the beta weight.
            gamma_constraint: Optional constraint for the gamma weight.
            beta_constraint: Optional constraint for the beta weight.
            kwargs: keyword arguments passed to :meth:`super().__init__`
        """
        super(LayerNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.center = center
        self.scale = scale
        if epsilon is None:
            epsilon = tf.keras.backend.epsilon() ** 2
        self.epsilon = epsilon
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma, self.beta = None, None

    def get_config(self) -> Mapping[str, Any]:
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'gamma_initializer': self.gamma_initializer,
            'beta_initializer': self.beta_initializer,
            'gamma_regularizer': self.gamma_regularizer,
            'beta_regularizer': self.beta_regularizer,
            'gamma_constraint': self.gamma_constraint,
            'beta_constraint': self.beta_constraint,
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        return input_shape

    def compute_mask(self, inputs, input_mask: Optional[Union[tf.Tensor, list]] = None) -> Union[tf.Tensor, list]:
        return input_mask

    def build(self, input_shape: tf.TensorShape) -> None:
        self.input_spec = tf.keras.engine.InputSpec(shape=input_shape)
        shape = input_shape[-1:]
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                name='gamma',
            )
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                name='beta',
            )
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        mean = tf.keras.backend.mean(inputs, axis=-1, keepdims=True)
        variance = tf.keras.backend.mean(tf.keras.backend.square(inputs - mean), axis=-1, keepdims=True)
        std = tf.keras.backend.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs
