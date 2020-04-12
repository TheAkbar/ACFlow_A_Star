import tensorflow as tf
import numpy as np
import pickle

class Dataset(object):
    def __init__(self, dfile, split, batch_size):
        super().__init__()
        
        with open(dfile, 'rb') as f:
            data_dict = pickle.load(f, encoding='latin1')
        data = data_dict[split]

        self.size = data.shape[0]
        self.d = data.shape[1]
        self.num_steps = self.size // batch_size

        dst = tf.data.Dataset.from_tensor_slices(data)
        if split == 'train':
            dst = dst.shuffle(self.size)
        dst = dst.batch(batch_size, drop_remainder=True)
        dst = dst.prefetch(1)
        dst_it = dst.make_initializable_iterator()
        x = dst_it.get_next()
        self.x = tf.reshape(x, [batch_size, self.d])
        self.y = tf.zeros([batch_size], dtype=tf.float32)
        self.m = tf.cast(tf.ones([batch_size, self.d]), tf.float32)
        self.b = 1 - tf.cast(tf.one_hot(np.random.randint(self.d, size=batch_size), self.d), tf.float32)
        # self.b = 1 - tf.cast(tf.one_hot(5, self.d), tf.float32)
        self.b = self.b * self.m
        self.initializer = dst_it.initializer

    def initialize(self, sess):
        sess.run(self.initializer)

