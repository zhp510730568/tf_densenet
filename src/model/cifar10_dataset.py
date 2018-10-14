import tensorflow as tf
from tensorflow.python.client.session import Session
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.ops.control_flow_ops import cond

from src.model.base_dataset import BaseDataset

def map_fn(x, y):
    '''
    data preprocess
    :param x:
    :param y:
    :return:
    '''
    x = tf.cast(x, dtype=tf.float32) / 255.0
    y = tf.cast(y, dtype=tf.int32)
    return x, y


class Cifar10Dataset(BaseDataset):
    '''

    '''
    def __init__(self, batch_size=128):
        self._batch_size=batch_size
        (self._train_x, self._train_y), (self._test_x, self._test_y) = cifar10.load_data()

    def get_train_dataset(self, name_scope='train_dataset'):
        return self._get_dataset(self._train_x, self._train_y, name_scope)

    def get_test_dataset(self, name_scope='test_dataset'):
        return self._get_dataset(self._test_x, self._test_y, name_scope)

    def _get_dataset(self, x, y, name_scope='data'):
        '''
        get dataset
        :param name_scope:
        :param x:
        :param y:
        :return: train_init_op, train_next_op
        '''
        batch_size = self._batch_size
        with tf.name_scope(name=name_scope):
            dataset = Dataset.from_tensor_slices((x, y))
            dataset = dataset.shuffle(buffer_size=batch_size * 10) \
                .repeat() \
                .map(map_func=map_fn, num_parallel_calls=8) \
                .batch(batch_size=batch_size) \
                .prefetch(buffer_size=batch_size * 10)

            init_op = dataset.make_initializable_iterator(shared_name='init_op')
            next_op = init_op.get_next(name='next_op')
        return init_op, next_op


if __name__=='__main__':
    is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
    dataset = Cifar10Dataset(batch_size=128)
    train_init_op, train_next_op = dataset.get_train_dataset()
    test_init_op, test_next_op = dataset.get_test_dataset()
    print(train_next_op)
    print(test_next_op)
    input = cond(is_training, true_fn=lambda: train_next_op, false_fn=lambda: test_next_op, name='input')

    import time
    with Session() as sess:
        sess.run([train_init_op.initializer, test_init_op.initializer])
        start_time = time.time()
        for _ in range(1000):
            input_value=sess.run([input], feed_dict={is_training: True})
            print(input_value[0][0].shape)
        print('total time: %d' % (time.time() - start_time))