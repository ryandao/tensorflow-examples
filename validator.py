from __future__ import division
from __future__ import print_function

import sys
from mnist_network import MNISTNetwork
import tensorflow as tf
from PIL import Image

path = sys.argv[1]
img = Image.open(path).convert('LA')
data = [x[1] / 255 for x in img.getdata()]

with tf.Session() as session:
    network = MNISTNetwork([784, 30, 10])
    network.load(session, 'model.ckpt')
    print("Network prediction: %s" % network.predict(session, [data])[0])
