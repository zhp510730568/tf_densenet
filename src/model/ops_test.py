import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow.python.framework import ops

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables

from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer
from tensorflow.python.ops import gradients_impl

import tensorflow as tf

# We define our Variables and placeholders
x = tf.placeholder(tf.int32, shape=[], name='x')
y = tf.Variable(2, dtype=tf.int32)

# We set our assign op
assign_op = tf.assign(y, y + 1)
tf.add_to_collection(name=tf.GraphKeys.UPDATE_OPS, value=assign_op)

updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
print(updates)
for update in updates:
    print(update)

tf.summary.histogram('var', y)
# We build our multiplication (this could be a more complicated graph)
with tf.control_dependencies([assign_op]):
    print('Print')
    tf.Print(y, [x])
    out = x * y
# I define a "shape-able" Variable
x1 = tf.Variable(
    [],
    dtype=tf.int32,
    validate_shape=False, # By "shape-able", i mean we don't validate the shape
    trainable=False
)

merge = tf.summary.merge_all()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter('../logdir/', sess.graph)
    print(sess.run(x1))
    for i in range(3):
        hist, o = sess.run([merge, out], feed_dict={x: 1})
        saver.save(sess, 'ckpt/test.ckpt', global_step=i + 1)
        train_writer.add_summary(hist)