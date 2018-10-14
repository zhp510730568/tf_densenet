import numpy as np

import tensorflow as tf

from src.model.logger import Logger

logger = Logger()


class DenseNet(object):
    def __init__(self, growth_rate=32, is_training=False, compress_factor=0.5):
        self._growth_rate = growth_rate
        self._compress_factor = compress_factor
        self._is_training = is_training

    def _dense_block(self, input, block_layers=12, name='dense_block1'):
        with tf.name_scope(name=name):
            layers_concat = list()
            layers_concat.append(input)

            x = self._bottleneck_layer(input, scope=name + '_bottleN_' + str(0))

            layers_concat.append(x)

            for layer in range(block_layers - 1):
                name = 'block_layer_%d' % (layer + 1)
                x = tf.concat(layers_concat, axis=3)
                x = self._bottleneck_layer(x, scope=name)
                layers_concat.append(x)

            x = tf.concat(layers_concat, axis=3)

            return x

    @staticmethod
    def _conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
        with tf.name_scope(layer_name):
            network = tf.layers.conv2d(inputs=input, filters=filter, kernel_size=kernel, strides=stride, padding='SAME')
            return network

    def _bottleneck_layer(self, input, scope='conv'):
        with tf.name_scope(name=scope):
            x = self._batch_norm(input=input, if_training=self._is_training)
            x = tf.nn.relu(x)
            x = tf.layers.conv2d(x, filters=self._growth_rate * 4, kernel_size=[1, 1], strides=1, padding='SAME')
            x = tf.layers.dropout(x, rate=0.2, training=self._is_training)

            x = self._batch_norm(x, if_training=self._is_training)
            x = tf.nn.relu(x)
            x = tf.layers.conv2d(x, filters=self._growth_rate, kernel_size=[3, 3], strides=1, padding='SAME')
            x = tf.layers.dropout(x, rate=0.2, training=self._is_training)
            return x

    def _transition_layer(self, input, name='transition_layer'):
        with tf.name_scope(name=name):
            x = self._batch_norm(input=input, if_training=self._is_training)
            x = tf.nn.relu(x)
            x = tf.layers.conv2d(x, filters=self._growth_rate, kernel_size=[1, 1], strides=1, padding='SAME')
            x = tf.layers.dropout(x, rate=0.2, training=self._is_training)
            x = tf.layers.average_pooling2d(x, pool_size=[2, 2], strides=2, padding='VALID')

            return x

    def _batch_norm(self, input, if_training):
        """
        Batch normalization on convolutional feature maps.
        Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
        Args:
            input:                  Tensor, 4D NHWC input feature maps
            depth:                  Integer, depth of input feature maps
            if_training:            Boolean tf.Varialbe, true indicates training phase
            scope:                  String, variable scope
        Return:
            normed_tensor:          Batch-normalized feature maps
        """
        with tf.variable_scope('batch_normalization'):
            depth = int(input.get_shape()[-1])
            beta = tf.Variable(tf.constant(0.0, shape=[depth]),
                               name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[depth]),
                                name='gamma', trainable=True)
            batch_mean, batch_var = tf.nn.moments(input, [0, 1, 2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.99)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(if_training,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed_tensor = tf.nn.batch_normalization(input, mean, var, beta, gamma, 1e-3)

        return normed_tensor

    @staticmethod
    def _global_average_pooling(x, stride=1):
        width = np.shape(x)[1]
        height = np.shape(x)[2]
        pool_size = [width, height]

        return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride)

    def desnet(self, input):
        with tf.device('/GPU:0'):
            x = self._conv_layer(input, filter=2 * self._growth_rate, kernel=[7,7], stride=2, layer_name='conv0')
            x = tf.layers.max_pooling2d(inputs=x, pool_size=[3,3], strides=[2, 2], padding='VALID')

            for i in range(2):
                x = self._dense_block(input=x, block_layers=self._growth_rate, name='dense_'+str(i))
                x = self._transition_layer(x, name='trans_'+str(i))

            x = self._dense_block(input=x, block_layers=self._growth_rate, name='dense_final')

            x = self._batch_norm(x, if_training=self._is_training)
            x = tf.nn.relu(x)
            x = self._global_average_pooling(x)

            x = tf.layers.flatten(x)

            x = tf.layers.dense(x, units=10, name='linear')

        return x