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

from typing import Union, Optional

import tensorflow as tf


class Vocabulary:
    """
    Experimental vocabulary built on top of tf.data and TF lookup ops. It is designed to be passed to tf.keras.Model
    initialization in order to work as part of the corresponding TF Graph.

    Args:
        source: either string that contains the path to source file, or tf.Tensor. See``mapping`` argument of
            tf.contrib.lookup.index_table_from_file or lookup.index_to_string_table_from_file for details
        index_default_value: value returned in case of string not in vocabulary
        reverse_default_value value returned in case of index does not correspond to any string in vocabulary
    """
    def __init__(self,
                 source: Union[str, tf.Tensor],
                 index_default_value: int = 0,
                 reverse_default_value: str = '<UNK>'
                 ) -> None:
        if isinstance(source, str):
            self.lookup_table = tf.contrib.lookup.index_table_from_file(source, default_value=index_default_value)
            self.reverse_lookup_table = tf.contrib.lookup.index_to_string_table_from_file(
                source, default_value=reverse_default_value
            )
        elif isinstance(source, tf.Tensor):
            self.lookup_table = tf.contrib.lookup.index_table_from_tensor(source, default_value=index_default_value)
            self.reverse_lookup_table = tf.contrib.lookup.index_to_string_table_from_tensor(
                source, default_value=reverse_default_value
            )
        else:
            raise ValueError(f'{source} must be a string path to vocabulary file or tf.Tensor')

    @classmethod
    def from_dataset(cls, dataset: tf.data.Dataset, istarget: bool = False, key: Optional[str] = None) -> 'Vocabulary':
        """Constructs vocabulary from tf.data.Dataset"""
        raise NotImplementedError
        # return cls(source=tf.constant(['<PAD>', '<UNK>']))

    def __getitem__(self, items) -> tf.Tensor:
        """Probably it is bad pattern, however we need this for prototyping."""
        return self.lookup(items)

    def lookup(self, items: tf.Tensor) -> tf.Tensor:
        """..."""
        return self.lookup_table.lookup(items)

    def reverse_lookup(self, items: tf.Tensor) -> tf.Tensor:
        """..."""
        return self.reverse_lookup_table.lookup(items)

    def size(self) -> int:
        """..."""
        return int(self.lookup_table.size())
