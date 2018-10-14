import tensorflow as tf

tf.flags.DEFINE_integer(name='batch_size', default=128, help='--batch_size 128')
tf.flags.DEFINE_float(name='init_learning_rate', default=1e-3, help='--init_learning_rate 1e-1')
tf.flags.DEFINE_integer(name='total_epoches', default=90, help='--total_epoches 90')

FLAGS = tf.flags.FLAGS
print(FLAGS)
if __name__=='__main__':
    print(FLAGS.batch_size)
    print(FLAGS.init_learning_rate)
    print(FLAGS.total_epoches)