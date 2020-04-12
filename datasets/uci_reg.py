import tensorflow as tf
import numpy as np
import math
import pickle

def _parse_v1(i, x, y, d):
    b = np.random.binomial(1, 0.5, [d]).astype(np.float32)
    m = b.copy()
    w = list(np.where(b == 0)[0])
    w.append(-1)
    w = np.random.choice(w)
    if w >= 0:
        m[w] = 1.

    return i, x, y, b, m

def _parse_v2(i, x, y, d):
    b = np.zeros([d], dtype=np.float32)
    no = np.random.choice(d+1)
    o = np.random.choice(d, [no], replace=False)
    b[o] = 1.
    m = b.copy()
    w = list(np.where(b == 0)[0])
    w.append(-1)
    w = np.random.choice(w)
    if w >= 0:
        m[w] = 1.

    return i, x, y, b, m

class Dataset(object):
    def __init__(self, dfile, split, batch_size, mask):
        super().__init__()

        with open(dfile, 'rb') as f:
            data_dict = pickle.load(f)
        data, label = data_dict[split]
        
        self.size = data.shape[0]
        self.d = data.shape[1]
        self.num_steps = math.ceil(self.size / batch_size)

        if mask == 'v1':
            _parse = _parse_v1
        elif mask == 'v2':
            _parse = _parse_v2
        else:
            raise Exception()

        ind = tf.range(self.size, dtype=tf.int64)
        dst = tf.data.Dataset.from_tensor_slices((ind, data, label))
        if split == 'train':
            dst = dst.shuffle(self.size)
        dst = dst.map(lambda i, x, y: tuple(
            tf.py_func(_parse, [i, x, y, self.d], 
            [tf.int64, tf.float32, tf.float32, tf.float32, tf.float32])),
            num_parallel_calls=16)
        dst = dst.batch(batch_size)
        dst = dst.prefetch(1)
        dst_it = dst.make_initializable_iterator()
        self.i, self.x, self.y, self.b, self.m = dst_it.get_next()
        self.initializer = dst_it.initializer

    def initialize(self, sess):
        sess.run(self.initializer)