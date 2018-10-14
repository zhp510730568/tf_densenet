import tensorflow as tf

print(tf.__version__)
w1 = tf.Variable(2.0)
w2 = tf.Variable(2.0)

a = tf.multiply(w1, 3.0)
a_stoped = tf.stop_gradient(a)

# b=w1*3.0*w
b = tf.multiply(a_stoped, w2)
c = tf.multiply(b, w1)
tf.clip_by_value
gradients = tf.gradients(c, xs=[w1, w2])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(a_stoped))
    print(sess.run(b))
    print(sess.run(c))
    print(sess.run(gradients))

tf.get_variable