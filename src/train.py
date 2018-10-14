#! /usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np

import tensorflow as tf
from src.model.cifar10_dataset import Cifar10Dataset
from src.model.densenet import DenseNet

if __name__ == '__main__':
    cifar10 = tf.keras.datasets.cifar10
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()
    dataset = Cifar10Dataset(batch_size=128)

    data_input = tf.placeholder(tf.float32, (None, 32, 32, 3), 'data_input')
    y = tf.placeholder(tf.int32, (None), 'labels')
    labels = tf.one_hot(y, 10)

    is_training = tf.placeholder(dtype=tf.bool, name='training_flag')
    learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

    densetnet = DenseNet(growth_rate=32, is_training=is_training, compress_factor=0.5)
    logits = densetnet.desnet(data_input)

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
    mean_loss = tf.reduce_mean(loss)

    epsilon = 1e-8
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.99)
    train_op = optimizer.minimize(mean_loss)

    accuracy = tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1)), dtype=tf.float32)
    accuracy = tf.reduce_mean(accuracy)

    batch_size = 256
    init_learning_rate = 1e-2
    total_epochs = 90

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        epoch_learning_rate = init_learning_rate
        test_input = test_x[0:10000] / 255
        test_output = np.reshape(test_y[0: 10000], newshape=(10000))
        for epoch in range(0, 100):
            iterations = 50000 / batch_size
            if epoch == (total_epochs * 0.5) or epoch == (total_epochs * 0.75):
                epoch_learning_rate = epoch_learning_rate / 10
            for index in range(0, int(iterations)):
                input = train_x[index * batch_size: (index + 1) * batch_size] / 255.0
                output = np.reshape(train_y[index * batch_size: (index + 1) * batch_size], newshape=(batch_size))

                train_feed_dict = {
                    data_input: input,
                    y: output,
                    is_training: True,
                    learning_rate: epoch_learning_rate
                }
                sess.run([train_op], feed_dict=train_feed_dict)
                if index % 100 == 0:
                    test_feed_dict = {
                        data_input: test_input,
                        y: test_output,
                        is_training: False
                    }
                    test_logits, loss_value, test_accuracy = sess.run([logits, mean_loss, accuracy], feed_dict=test_feed_dict)
                    print('epoch: %d, loss: %f, test accuracy: %f' % (epoch, loss_value, test_accuracy))