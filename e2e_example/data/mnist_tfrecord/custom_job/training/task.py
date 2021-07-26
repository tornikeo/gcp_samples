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

import argparse

import tensorflow as tf


def make_and_compile_model(learning_rate):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(filters=2, kernel_size=3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=10, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy())

    return model


def make_tf_dataset(train_data_gcs_uri):
    raw_ds = tf.data.TFRecordDataset(train_data_gcs_uri)

    # See the SerializeToString call in mnist_to_tfrecord.py.
    def parse(serialized):
        feature_description = {
            'label': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string)
        }

        return tf.io.parse_single_example(serialized, feature_description)

    # See the tobytes call in mnist_to_tfrecord.py.
    def decode(parsed):
        flattened_image = tf.io.decode_raw(parsed['image_raw'], out_type=tf.uint8)
        image = tf.reshape(flattened_image, (28, 28, 1))

        return {'image': image, 'label': parsed['label']}

    def transform(decoded):
        image = decoded['image']
        feature = tf.cast(image, tf.float32) / 255.0

        label = decoded['label']
        target = tf.one_hot(label, depth=10)

        return (feature, target)


    parsed_ds = raw_ds.map(parse)
    decoded_ds = parsed_ds.map(decode)
    ds = decoded_ds.map(transform)

    return ds


def main(args):
    dataset = make_tf_dataset(args.train_data_gcs_uri)
    model = make_and_compile_model(args.learning_rate)

    model.fit(
        x=dataset.shuffle(1024).batch(args.batch_size),
        epochs=args.epochs
    )

    # Define a serving signature to accept image raw bytes.
    # Note that the same code is also used in make_tf_dataset.
    @tf.function
    def serve(image_raw, key):
        flattened_image = tf.io.decode_raw(image_raw, out_type=tf.uint8)
        image = tf.reshape(flattened_image, (1, 28, 28, 1))
        feature = tf.cast(image, tf.float32) / 255.0

        output = model(feature)

        return {'scores': output, 'key': key}

    signature = serve.get_concrete_function(image_raw=tf.TensorSpec([None,], dtype=tf.string), key=tf.TensorSpec([None,], dtype=tf.string))

    model.save(
        filepath=args.model_dir,
        save_format='tf',
        # This would be added under the `serving_default` signature key.
        signatures=signature
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_data_gcs_uri', default='gs://cloud-samples-data/ai-platform/mnist_tfrecord/train.tfrecord', type=str)
    parser.add_argument('--model_dir', default='/tmp/mnist', type=str)

    parser.add_argument('--learning_rate', default=0.1, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=2, type=int)

    args = parser.parse_args()

    main(args)
