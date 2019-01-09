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

from typing import Optional, Tuple

import tensorflow as tf


DEFAULT_LABEL_INDEX = 0


label_vocab = tf.contrib.lookup.index_table_from_tensor(tf.constant(['skip',
                                                                     'speech',
                                                                     'neutral',
                                                                     'negative',
                                                                     'positive']),
                                                        default_value=DEFAULT_LABEL_INDEX)


def train_input_fn(batch_size: int = 32,
                   tokenizer: Optional[callable] = None,
                   shuffle: bool = True,
                   epochs: Optional[int] = None,
                   prefetch: Optional[int] = None
                   ) -> tf.data.Dataset:
    """
    Download train data (both random and preselected posts) from https://github.com/text-machine-lab/rusentiment,
    transform it and produce dataset in TensorFlow format.

    Todo:
        think of proper shuffling and train/dev split
    """
    dataset = _extract()

    dataset = dataset.map(_map_fn)

    if tokenizer is None:
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
    else:
        raise NotImplementedError('Tokenization is not implemented yet for this dataset')

    if shuffle:
        dataset = dataset.shuffle(buffer_size=8*batch_size)

    if epochs is not None:
        dataset = dataset.repeat(count=epochs)

    if prefetch is not None:
        dataset.prefetch(buffer_size=prefetch)

    return dataset


def test_input_fn(batch_size: int = 32,
                  tokenizer: Optional[callable] = None,
                  prefetch: Optional[int] = None
                  ) -> tf.data.Dataset:
    """
    Download test data from https://github.com/text-machine-lab/rusentiment, transform it and produce dataset in
    TensorFlow format.
    """
    dataset = _extract(data_type='test')

    dataset = dataset.map(_map_fn)

    if tokenizer is None:
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
    else:
        raise NotImplementedError('Tokenization is not implemented yet for this dataset')

    if prefetch is not None:
        dataset.prefetch(buffer_size=prefetch)

    return dataset


def _extract(data_type: str = 'train') -> tf.data.Dataset:
    """Download data from the web or cache and output tf.data.Dataset ready to be transformed."""

    base_url = 'https://raw.githubusercontent.com/text-machine-lab/rusentiment/master/Dataset/{}'

    if data_type == 'train':

        random_data_path = tf.keras.utils.get_file(fname='rusentiment_random_posts.csv',
                                                   origin=base_url.format('rusentiment_random_posts.csv'))
        preselected_data_path = tf.keras.utils.get_file(fname='rusentiment_preselected_posts.csv',
                                                        origin=base_url.format('rusentiment_preselected_posts.csv'))

        train_ds = tf.data.experimental.CsvDataset(random_data_path, record_defaults=[tf.string, tf.string])
        additional_train_ds = tf.data.experimental.CsvDataset(preselected_data_path, record_defaults=[tf.string, tf.string])

        return train_ds.concatenate(additional_train_ds)

    elif data_type == 'test':

        test_data_path = tf.keras.utils.get_file(fname='rusentiment_tests.csv',
                                                 origin=base_url.format('rusentiment_tests.csv'))

        return tf.data.experimental.CsvDataset(test_data_path, record_defaults=[tf.string, tf.string])

    else:
        raise ValueError(f'Unrecognized data type {data_type}! Please choose either "train" or "test"')


# TODO: fix issues with type annotation
# def _map_fn(l: tf.string, t: tf.string) -> Tuple[tf.string, tf.int64]:
def _map_fn(l, t):
    """
    This looks like dataset-specific stuff, so we don't make it reusable across different datasets.

    Note:
        This mapping is considered stateful, so it is possible to iterate over resulting dataset only with
        initializable(*) iterators. In addition, in graph mode you probably need to run tf.tables_initializer().
    """
    nl = label_vocab.lookup(l)
    return t, nl
