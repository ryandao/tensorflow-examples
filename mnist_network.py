from __future__ import division
from __future__ import print_function

import tensorflow as tf
import json
from tensorflow.examples.tutorials.mnist import input_data

class MNISTNetwork:
    def __init__(self, sizes):
        self.sizes = sizes
        self.weights = [tf.Variable(tf.random_normal([x, y]), name='weights') for (x, y) in zip(sizes[:-1], sizes[1:])]
        self.biases = [tf.Variable(tf.random_normal([x]), name='biases') for x in sizes[1:]]

    def predict(self, session, inputs):
        return session.run(tf.argmax(tf.nn.softmax(self.logits(inputs)), 1))

    def logits(self, inputs):
        last_weight = self.weights[-1]
        last_bias = self.biases[-1]
        logits = inputs

        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            logits = tf.sigmoid(tf.matmul(logits, weight) + bias)

        return tf.matmul(logits, last_weight) + last_bias

    def train(self):
        mnist = input_data.read_data_sets('data', one_hot=True)
        inputs = tf.placeholder(tf.float32, [None, 784], name='inputs')
        labels = tf.placeholder(tf.float32, [None, 10], name='labels')
        logits = self.logits(inputs)
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, labels))
        train_step = tf.train.GradientDescentOptimizer(3.0).minimize(cost)

        with tf.Session() as session:
            saver = tf.train.Saver() # For saving the result graph
            summary_writer = tf.train.SummaryWriter('.', session.graph) # For graph visualization
            session.run(tf.initialize_all_variables())

            for i in range(30):
                while not mnist.train.epochs_completed:
                    batch_xs, batch_ys = mnist.train.next_batch(10)
                    session.run(train_step, {inputs: batch_xs, labels: batch_ys})
                mnist.train._epochs_completed = 0
                mnist.train._index_in_epoch = 0

                # Test trained model
                correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                print("Epoch %s:" % i, accuracy.eval({inputs: mnist.test.images, labels: mnist.test.labels}))

            save_path = saver.save(session, "model.ckpt")
            print("Model saved to %s" % save_path)
            summary_writer.close()

    def load(self, session, path):
        saver = tf.train.Saver()
        saver.restore(session, path)

if __name__ == '__main__':
    network = MNISTNetwork([784, 30, 10])
    network.train()
