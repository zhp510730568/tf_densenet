import numpy as np

import tensorflow as tf
from tensorflow.python.util.tf_export import tf_export

width = 10
height = 10
channel = 1
vector_size=100
time_steps = 10

input=tf.placeholder(dtype=tf.float32, shape=[None, time_steps, width, height, channel])

input=tf.placeholder(dtype=tf.float32, shape=[None, 1000])

hidden_layer = tf.get_variable(shape=[vector_size], dtype=tf.float32, name='hidden_weight', initializer=tf.zeros_initializer)

W = tf.get_variable(shape=[1000, vector_size], dtype=tf.float32, name='x2h_weight')
U = tf.get_variable(shape=[vector_size, vector_size], dtype=tf.float32, name='h2h_weight')


def conv_net(input, name_scope='conv_net'):
    with tf.variable_scope(name_or_scope=name_scope, reuse=tf.AUTO_REUSE):
        with tf.device('/GPU:0'):
            output = tf.layers.conv2d(input, kernel_size=1, filters=10, strides=(1, 1), padding="SAME", name='conv2d_layer1')
            output = tf.layers.conv2d(output, kernel_size=1, filters=10, strides=(1, 1), padding="SAME", name='conv2d_layer2')
            output = tf.layers.conv2d(output, kernel_size=1, filters=10, strides=(1, 1), padding="SAME", name='conv2d_layer3')
            output = tf.layers.max_pooling2d(output, pool_size=2, strides=(2, 2), padding="SAME", name='conv2d_pool1')

            output = tf.layers.conv2d(output, kernel_size=1, filters=10, strides=(1, 1), padding="SAME", name='conv2d_layer4')
            output = tf.layers.conv2d(output, kernel_size=1, filters=10, strides=(1, 1), padding="SAME", name='conv2d_layer5')
            output = tf.layers.conv2d(output, kernel_size=1, filters=10, strides=(1, 1), padding="SAME", name='conv2d_layer6')
            output = tf.layers.max_pooling2d(output, pool_size=2, strides=(2, 2), padding="SAME", name='conv2d_pool2')

            output = tf.layers.flatten(output, name='conv2d_flatten')

            return output

for _ in range(time_steps):
    hidden_layer = tf.tanh(tf.matmul(input, W) + tf.matmul(hidden_layer, U))

tf.AUTO_REUSE
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    random_input = np.random.uniform(0, 1.0, (1, 1000))
    for _ in range(1000):
        data_output = sess.run(hidden_layer, feed_dict={input: random_input})
        print(data_output)