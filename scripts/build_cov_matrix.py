import os
import sys
p = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(p)
import argparse
import logging
import tensorflow as tf
import numpy as np
import pickle
from pprint import pformat, pprint

from datasets import get_dataset
from models import get_model
from utils.hparams import HParams

def dataset(params, point, std=100, num_samps=4):
    point = np.expand_dims(point, 0)
    t_dataset = []
    masks = []
    for d in range(point.shape[1]):
        t_dataset.append(point)
        mask = np.ones_like(point)
        mask[:, d] = 0
        masks.append(mask)
        for i in range(point.shape[1]):
            p = np.tile(point, (num_samps, 1))
            if i != d:
                p[:, i] += std * (np.random.rand(num_samps) - 0.5)
            t_dataset.append(p)
            mask = np.ones_like(p)
            mask[:, d] = 0
            masks.append(mask)
    t_dataset.append(point)
    masks.append(np.zeros_like(point))
    return np.vstack(t_dataset), np.vstack(masks)


parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str)
args = parser.parse_args()

params = HParams(args.cfg_file)
pprint(params.dict)

os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu
np.random.seed(params.seed)
tf.set_random_seed(params.seed)

############################################################
logging.basicConfig(filename=params.exp_dir + '/test.log',
                    filemode='w',
                    level=logging.INFO,
                    format='%(message)s')
logging.info(pformat(params.dict))
############################################################

testset = get_dataset('test', params)

x_ph = tf.placeholder(tf.float32, [None, params.dimension])
b_ph = tf.placeholder(tf.float32, [None, params.dimension])
m_ph = tf.placeholder(tf.float32, [None, params.dimension])
y_ph = tf.placeholder(tf.float32, [None])

model = get_model(params)
model.build(x_ph, y_ph, b_ph, m_ph)

############################################################

config = tf.ConfigProto()
config.log_device_placement = False
config.allow_soft_placement = True
# config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

saver = tf.train.Saver(tf.global_variables())
weights_dir = os.path.join(params.exp_dir, 'weights')
logging.info(f'Restoring parameters from {weights_dir}')
restore_from = tf.train.latest_checkpoint(weights_dir)
saver.restore(sess, restore_from)
np.set_printoptions(linewidth=200)

###########################################################4

# testset.initialize(sess)
# num_batches = testset.num_steps
# num_samps = 4
# tries = 100
# with open(params.dfile, 'rb') as f:
#     data = pickle.load(f)['test']
# avg_cov = np.zeros((params.dimension, params.dimension))
# for point in np.random.permutation(data)[:tries]:
#     x, b = dataset(params, point, num_samps=num_samps)
#     m = np.ones_like(b)
#     ll = sess.run(model.log_likel, {x_ph:x.astype(np.float32), b_ph:b, m_ph:m})
#     # test_ll.append(ll)
#     # test_ll = np.concatenate(test_ll, axis=0)
#     # test_ll = np.mean(test_ll)
#     cov = []
#     for d in np.split(ll[:-1], x.shape[1]):
#         true = d[0]
#         vals = np.array([np.mean(arr) for arr in np.split(d[1:], x.shape[1])])
#         cov.append(np.abs(vals-true))
#     print(np.array(cov))
#     avg_cov += np.array(cov)
#     logging.info(f'test log likelihood: {ll}')
# print(avg_cov/tries)


graph = tf.get_default_graph()
sparse_mat = graph.get_tensor_by_name('sparsity_mat:0').eval(session=sess)
sparse_mat = np.triu(sparse_mat) - np.diag(np.diag(sparse_mat))
sparse_mat += sparse_mat.T
print(sparse_mat)
print(1000*sparse_mat)