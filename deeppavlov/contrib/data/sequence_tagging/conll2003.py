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
from typing import Generator
from functools import partial

import tensorflow as tf

label_lookup_table = tf.contrib.lookup.index_table_from_file(
    vocabulary_file='/root/.keras/models/ner_conll2003/tag.dict',
    num_oov_buckets=0,
    vocab_size=None,
    default_value=0,
    hasher_spec=tf.contrib.lookup.FastHashSpec,
    key_dtype=tf.string,
    name=None,
    key_column_index=0,
    value_column_index=tf.contrib.lookup.TextFileIndex.LINE_NUMBER,
    delimiter='\t'
)


def train_input_fn(use_pos: bool = False) -> tf.data.Dataset:
    """"""

    dataset = _extract('train', use_pos=use_pos)

    dataset.map()

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

    output_types = ((tf.string, tf.string), tf.string) if use_pos else (tf.string, tf.string, tf.string)

    return tf.data.Dataset.from_generator(generator=data_gen, output_types=output_types)


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
                        samples.append(((tokens, pos_tags), tags))
                    else:
                        samples.append((tokens, tags))
                    yield samples[-1]
                    tokens = []
                    pos_tags = []
                    tags = []
            elif len(line) < 2:
                if len(tokens) > 0:
                    if use_pos:
                        samples.append(((tokens, pos_tags), tags))
                    else:
                        samples.append((tokens, tags,))
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

def map_fn():
    """..."""
    return
