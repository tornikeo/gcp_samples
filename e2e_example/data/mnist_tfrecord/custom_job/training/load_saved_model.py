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

import numpy as np
import tensorflow as tf

model_dir = '/tmp/mnist'
model_dir = 'gs://ucaip-samples-us-central1/mnist_tfrecord/pretrained'
model_dir = 'gs://cloud-samples-data-us-central1/ai-platform/mnist_tfrecord/pretrained'

loaded = tf.saved_model.load(model_dir)
wrapped_func = loaded.signatures['serving_default']

# The same format as loaded by tensorflow_datasets.
image_raw = np.random.randint(low=0, high=2, size=(1, 28, 28, 1), dtype=np.uint8).tobytes()

print(wrapped_func(image_raw=tf.constant(image_raw), key=tf.constant('mykey')))
