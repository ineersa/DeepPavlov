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

from pathlib import Path
from typing import Optional, Tuple, Callable, Mapping

import tensorflow as tf

from deeppavlov.contrib.vocabulary import Vocabulary


def make_label_indexer(
        tag_vocab: Vocabulary = Vocabulary(
            source=tf.constant(['skip', 'speech', 'neutral', 'negative', 'positive']), reverse_default_value='skip')
) -> Callable[[Mapping[str, tf.Tensor], tf.Tensor], Tuple[Mapping[str, tf.Tensor], tf.Tensor]]:
    """..."""

    def label_indexer(sample, label):
        return sample, tag_vocab.lookup(label)

    return label_indexer


def train_input_fn(batch_size: int = 32,
                   tokenizer: Optional[callable] = None,
                   shuffle: int = 32 * 8,
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

    dataset = dataset.map(make_label_indexer())

    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle)

    if tokenizer is None:
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
    else:
        raise NotImplementedError('Tokenization is not implemented yet for this dataset')

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

    dataset = dataset.map(make_label_indexer())

    if tokenizer is None:
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
    else:
        raise NotImplementedError('Tokenization is not implemented yet for this dataset')

    if prefetch is not None:
        dataset.prefetch(buffer_size=prefetch)

    return dataset


def _extract(data_type: str = 'train') -> tf.data.Dataset:
    """Download data from the web or cache and output tf.data.Dataset ready to be transformed."""

    # base_url = 'https://raw.githubusercontent.com/text-machine-lab/rusentiment/master/Dataset/{}'

    data_archive = tf.keras.utils.get_file(fname='rusentiment.tar.gz',
                                           origin='http://files.deeppavlov.ai/datasets/rusentiment.tar.gz',
                                           extract=True,
                                           cache_subdir='datasets')

    data_dir = Path(data_archive).resolve().parent / 'rusentiment'

    if data_type == 'train':

        random_data_path = str(data_dir / 'rusentiment_random_posts.csv')
        # tf.keras.utils.get_file(fname='rusentiment_random_posts.csv',
        #                                        origin=base_url.format('rusentiment_random_posts.csv'))

        preselected_data_path = str(data_dir / 'rusentiment_preselected_posts.csv')
        # tf.keras.utils.get_file(fname='rusentiment_preselected_posts.csv',
        #                                             origin=base_url.format('rusentiment_preselected_posts.csv'))

        train_ds = tf.data.experimental.CsvDataset(random_data_path, record_defaults=[tf.string, tf.string])
        additional_train_ds = tf.data.experimental.CsvDataset(preselected_data_path, record_defaults=[tf.string, tf.string])

        dataset = train_ds.concatenate(additional_train_ds)
        return dataset.map(lambda l, s: (s, l))

    elif data_type == 'test':

        test_data_path = str(data_dir / 'rusentiment_tests.csv')
        # tf.keras.utils.get_file(fname='rusentiment_tests.csv',
        #                                      origin=base_url.format('rusentiment_tests.csv'))

        dataset = tf.data.experimental.CsvDataset(test_data_path, record_defaults=[tf.string, tf.string])
        return dataset.map(lambda l, s: (s, l))

    else:
        raise ValueError(f'Unrecognized data type {data_type}! Please choose either "train" or "test"')
