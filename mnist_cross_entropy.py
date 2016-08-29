# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

with tf.Session() as session:
    # Create the model
    inputs = tf.placeholder(tf.float32, [None, 784])

    w_h = tf.Variable(tf.random_normal([784, 30], stddev=1))
    b_h = tf.Variable(tf.random_normal([30], stddev=1))

    w_o = tf.Variable(tf.random_normal([30, 10], stddev=1))
    b_o = tf.Variable(tf.random_normal([10], stddev=1))

    # Define loss and optimizer
    logits_h = tf.sigmoid(tf.matmul(inputs, w_h) + b_h)
    logits_o = tf.matmul(logits_h, w_o) + b_o

    labels = tf.placeholder(tf.float32, [None, 10])
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits_o, labels))
    train_step = tf.train.GradientDescentOptimizer(3.0).minimize(cost)

    # Train
    session.run(tf.initialize_all_variables())
    for i in range(100):
        while not mnist.train.epochs_completed:
            batch_xs, batch_ys = mnist.train.next_batch(10)
            session.run(train_step, {inputs: batch_xs, labels: batch_ys})
        mnist.train._epochs_completed = 0
        mnist.train._index_in_epoch = 0

        # Test trained model
        correct_prediction = tf.equal(tf.argmax(logits_o, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(accuracy.eval({inputs: mnist.test.images, labels: mnist.test.labels}))
