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
from typing import Generator, Callable, Mapping, Tuple, Optional
from functools import partial

import tensorflow as tf

from deeppavlov.contrib.vocabulary import Vocabulary


def number_replacer(sample, label):
    sample['tokens'] = tf.strings.regex_replace(sample['tokens'], tf.constant('[0-9]'), tf.constant('1'))
    return sample, label


def sequence_length_counter(sample, label):
    sample['sequence_len'] = tf.size(sample['tokens'])
    return sample, label


def make_label_indexer(
        tag_vocab: Vocabulary = Vocabulary(
            tf.constant(['O', 'B-LOC', 'B-PER', 'B-ORG', 'I-PER', 'I-ORG', 'B-MISC', 'I-LOC', 'I-MISC']))
) -> Callable[[Mapping[str, tf.Tensor], tf.Tensor], Tuple[Mapping[str, tf.Tensor], tf.Tensor]]:
    """..."""

    def label_indexer(sample, label):
        return sample, tag_vocab.lookup(label)

    return label_indexer


def train_input_fn(use_pos: bool = False,
                   batch_size: int = 32,
                   tokenizer: Optional[callable] = None,
                   shuffle: Optional[int] = None,
                   epochs: Optional[int] = None,
                   prefetch: Optional[int] = None
                   ) -> tf.data.Dataset:
    """..."""

    dataset = _extract('train', use_pos=use_pos)

    dataset = dataset.map(number_replacer).map(make_label_indexer()).map(sequence_length_counter)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle)

    # weird control flow... it was not possible to define variables for `padded_shapes` and `padding_value` and then
    # pass it to `padded_batch` method; instead I had to do it directly
    if tokenizer is None and use_pos:
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=({'tokens': tf.Dimension(None), 'pos_tags': tf.Dimension(None), 'sequence_len': ()},
                           tf.Dimension(None)),
            padding_values=({'tokens': '', 'pos_tags': 'X', 'sequence_len': tf.constant(0)},
                            tf.constant(0, dtype=tf.int64)),
            drop_remainder=False)
    elif tokenizer is None:
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=({'tokens': tf.Dimension(None), 'sequence_len': ()}, tf.Dimension(None)),
            padding_values=({'tokens': '', 'sequence_len': tf.constant(0)}, tf.constant(0, dtype=tf.int64)),
            drop_remainder=False)
    else:
        raise NotImplementedError('Tokenization is not supported for this dataset')

    if epochs is not None:
        dataset = dataset.repeat(count=epochs)

    if prefetch is not None:
        dataset.prefetch(buffer_size=prefetch)

    return dataset


def _extract(data_type: str = 'train', use_pos: bool = False) -> tf.data.Dataset:
    """Download data from the web or cache and output tf.data.Dataset ready to be transformed."""
    data_archive = tf.keras.utils.get_file(fname='conll2003.tar.gz',
                                           origin='http://files.deeppavlov.ai/deeppavlov_data/conll2003_v2.tar.gz',
                                           extract=True,
                                           cache_subdir='datasets/conll2003')

    data_dir = Path(data_archive).resolve().parent

    if data_type == 'train':
        file_path = data_dir / 'train.txt'
    elif data_type == 'valid':
        file_path = data_dir / 'valid.txt'
    elif data_type == 'test':
        file_path = data_dir / 'test.txt'
    else:
        raise ValueError(f'{data_type} does not exist for CONLL2003 dataset. Choose from "train", "valid", "test".')

    data_gen = partial(_parse_conll2003_file, file_path, use_pos)

    # output_types = ({'tokens': tf.string,
    #                  'pos_tags': tf.string}, tf.string) if use_pos else ({'tokens': tf.string}, tf.string)

    return tf.data.Dataset.from_generator(generator=data_gen, output_types=({'tokens': tf.string}, tf.string)) #output_types)


def _parse_conll2003_file(file_path: Path, use_pos: bool = False) -> Generator[list, None, None]:
    samples = []
    with file_path.open(encoding='utf8') as f:
        tokens = ['<DOCSTART>']
        pos_tags = ['O']
        tags = ['O']
        for line in f:
            # Check end of the document
            if 'DOCSTART' in line:
                if len(tokens) > 1:
                    if use_pos:
                        samples.append(({'tokens': tokens, 'pos_tags': pos_tags}, tags))
                    else:
                        samples.append(({'tokens': tokens}, tags))
                    yield samples[-1]
                    tokens = []
                    pos_tags = []
                    tags = []
            elif len(line) < 2:
                if len(tokens) > 0:
                    if use_pos:
                        samples.append(({'tokens': tokens, 'pos_tags': pos_tags}, tags))
                    else:
                        samples.append(({'tokens': tokens}, tags))
                    yield samples[-1]
                    tokens = []
                    pos_tags = []
                    tags = []
            else:
                if use_pos:
                    token, pos, *_, tag = line.split()
                    pos_tags.append(pos)
                else:
                    token, *_, tag = line.split()
                tags.append(tag)
                tokens.append(token)
