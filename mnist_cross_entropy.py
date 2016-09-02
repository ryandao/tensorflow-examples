from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

class MNISTNetwork:
    def __init__(self, sizes):
        self.sizes = sizes
        self.weights = [tf.Variable(tf.random_normal([x, y])) for (x, y) in zip(sizes[:-1], sizes[1:])]
        self.biases = [tf.Variable(tf.random_normal([x])) for x in sizes[1:]]

    def logit(self, inputs):
        last_weight = self.weights[-1]
        last_bias = self.biases[-1]
        logit = inputs

        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            logit = tf.sigmoid(tf.matmul(logit, weight) + bias)

        return tf.matmul(logit, last_weight) + last_bias

    def train(self):
        inputs = tf.placeholder(tf.float32, [None, 784])
        labels = tf.placeholder(tf.float32, [None, 10])
        logit = self.logit(inputs)
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logit, labels))
        train_step = tf.train.GradientDescentOptimizer(3.0).minimize(cost)
        mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

        with tf.Session() as session:
            session.run(tf.initialize_all_variables())
            for i in range(100):
                while not mnist.train.epochs_completed:
                    batch_xs, batch_ys = mnist.train.next_batch(10)
                    session.run(train_step, {inputs: batch_xs, labels: batch_ys})
                mnist.train._epochs_completed = 0
                mnist.train._index_in_epoch = 0

                # Test trained model
                correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(labels, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                print(accuracy.eval({inputs: mnist.test.images, labels: mnist.test.labels}))

network = MNISTNetwork([784, 30, 10])
network.train()
