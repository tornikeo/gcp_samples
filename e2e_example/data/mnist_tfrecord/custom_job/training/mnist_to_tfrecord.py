# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import tensorflow_datasets as tfds


def mnist_to_tfexample(item):
    # An item is as loaded by tfds: a dict with keys 'image' and 'label'.
    image_raw = item['image'].numpy().tobytes()
    label = item['label'].numpy()

    feature = {
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def mnist_ds_to_tfrecord(mnist_ds, filepath):
    with tf.io.TFRecordWriter(filepath) as writer:
        for item in mnist_ds:
            tfexample = mnist_to_tfexample(item)
            writer.write(tfexample.SerializeToString())


def get_mnist_ds(split):
    return tfds.load('mnist', split=split)


if __name__ == '__main__':
    out_root = '/tmp/mnist'
    for split in ['train', 'test']:
        filepath = f'{out_root}/{split}.tfrecord'
        mnist_ds = get_mnist_ds(split)
        mnist_ds_to_tfrecord(mnist_ds, filepath)
