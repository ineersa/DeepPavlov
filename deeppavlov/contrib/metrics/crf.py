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
import tensorflow.python.keras.backend as K


def crf_nll(y_true, y_pred):
    """
    The negative log-likelihood for linear chain Conditional Random Field (CRF). This loss function is only used when
    the `layers.CRF` layer is trained in the "join" mode.

    Args:
        y_true: tensor with true targets.
        y_pred: tensor with predicted targets.

    Returns:
        A scalar representing corresponding to the negative log-likelihood.

    Raises:
        TypeError: If CRF is not the last layer.
    """
    crf, idx = y_pred._keras_history[:2]
    if crf._outbound_nodes:
        raise TypeError('When learn_model="join", CRF must be the last layer.')
    if crf.sparse_target:
        y_true = K.one_hot(K.cast(y_true[:, :, 0], 'int32'), crf.units)
    X = crf._inbound_nodes[idx].input_tensors[0]
    mask = crf._inbound_nodes[idx].input_masks[0]
    nloglik = crf.get_negative_log_likelihood(y_true, X, mask)
    return nloglik


def crf_loss(y_true, y_pred):
    """
    General CRF loss function depending on the learning mode.
    Args:
        y_true: tensor with true targets.
        y_pred: tensor with predicted targets.
    Returns
        If the CRF layer is being trained in the join mode, returns the negative log-likelihood. Otherwise returns the
        categorical crossentropy implemented by the underlying Keras backend.
    """
    crf, idx = y_pred._keras_history[:2]
    if crf.learn_mode == 'join':
        return crf_nll(y_true, y_pred)
    else:
        if crf.sparse_target:
            return tf.keras.metrics.sparse_categorical_crossentropy(y_true, y_pred)
        else:
            return tf.keras.metrics.categorical_crossentropy(y_true, y_pred)


def _get_accuracy(y_true, y_pred, mask, sparse_target=False):
    y_pred = K.argmax(y_pred, -1)
    if sparse_target:
        y_true = K.cast(y_true[:, :, 0], K.dtype(y_pred))
    else:
        y_true = K.argmax(y_true, -1)
    judge = K.cast(K.equal(y_pred, y_true), K.floatx())
    if mask is None:
        return K.mean(judge)
    else:
        mask = K.cast(mask, K.floatx())
        return K.sum(judge * mask) / K.sum(mask)


def crf_viterbi_accuracy(y_true, y_pred):
    """Use Viterbi algorithm to get best path, and compute its accuracy.`y_pred` must be an output from CRF."""
    crf, idx = y_pred._keras_history[:2]
    X = crf._inbound_nodes[idx].input_tensors[0]
    mask = crf._inbound_nodes[idx].input_masks[0]
    y_pred = crf.viterbi_decoding(X, mask)
    return _get_accuracy(y_true, y_pred, mask, crf.sparse_target)


def crf_marginal_accuracy(y_true, y_pred):
    """Use time-wise marginal argmax as prediction.`y_pred` must be an output from CRF with `learn_mode="marginal"`."""
    crf, idx = y_pred._keras_history[:2]
    X = crf._inbound_nodes[idx].input_tensors[0]
    mask = crf._inbound_nodes[idx].input_masks[0]
    y_pred = crf.get_marginal_prob(X, mask)
    return _get_accuracy(y_true, y_pred, mask, crf.sparse_target)


def crf_accuracy(y_true, y_pred):
    """Get default accuracy based on CRF `test_mode`."""
    crf, idx = y_pred._keras_history[:2]
    if crf.test_mode == 'viterbi':
        return crf_viterbi_accuracy(y_true, y_pred)
    else:
        return crf_marginal_accuracy(y_true, y_pred)
