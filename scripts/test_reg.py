import os
import sys
p = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(p)
import argparse
import logging
import tensorflow as tf
import numpy as np
import pickle
import gzip
from pprint import pformat, pprint
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from scipy.stats import entropy

from datasets import get_dataset
from models import get_model
from utils.hparams import HParams

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_samples', type=int, default=100)
args = parser.parse_args()

params = HParams(args.cfg_file)
params.gpu = args.gpu
params.batch_size = args.batch_size
params.num_samples = args.num_samples
pprint(params.dict)

os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu
np.random.seed(params.seed)
tf.set_random_seed(params.seed)

save_dir = os.path.join(params.exp_dir, 'cmi_greedy')
os.makedirs(save_dir, exist_ok=True)

############################################################
logging.basicConfig(filename=save_dir + '/cmi_greedy.log',
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
config.log_device_placement = True
config.allow_soft_placement = True
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

saver = tf.train.Saver(tf.global_variables())
weights_dir = os.path.join(params.exp_dir, 'weights')
logging.info(f'Restoring parameters from {weights_dir}')
restore_from = tf.train.latest_checkpoint(weights_dir)
saver.restore(sess, restore_from)

###########################################################

def batch_mse(s, y):
    '''
    s: [B,N] or [B]
    y: [B]
    '''
    rank = len(s.shape)
    if rank == 2:
        mse = (s - y[:,None]) ** 2
        mse = np.mean(mse, axis=1)
    elif rank == 1:
        mse = (s - y) ** 2

    return mse

def next_m(b):
    num = np.sum(b[0] == 0)
    for i in range(num):
        m = b.copy()
        for mi in m:
            d = np.where(mi == 0)[0]
            mi[d[i]] = 1
        yield m

def select_next(x, b):
    B = x.shape[0]
    N = params.num_samples
    d = params.dimension
    y = -np.ones([B], dtype=np.float32)

    best_cmi = -np.ones([B]) * np.inf
    best_mask = np.zeros_like(b)
    for m in next_m(b):
        sam_j = sess.run(model.sam_j, {x_ph:x, y_ph:y, b_ph:b, m_ph:m})
        sam_x = sam_j[:,:,:-1]
        sam_y = sam_j[:,:,-1]
        q = np.expand_dims(m*(1-b), axis=1)
        sam_x = sam_x * q + np.expand_dims(x, axis=1) * (1-q)
        sam_x = sam_x.reshape([B*N, d])
        sam_y = sam_y.reshape([B*N])
        b_tile = np.repeat(np.expand_dims(b, axis=1), N, axis=1).reshape([B*N, d])
        m_tile = np.repeat(np.expand_dims(m, axis=1), N, axis=1).reshape([B*N, d])
        log_ratio = sess.run(model.log_ratio, {x_ph:sam_x, y_ph:sam_y, b_ph:b_tile, m_ph:m_tile})
        log_ratio = log_ratio.reshape([B,N])
        cmi = np.mean(log_ratio, axis=1)
        Bid = cmi > best_cmi
        best_mask[Bid] = m[Bid].copy()
        best_cmi[Bid] = cmi[Bid].copy()

    return best_mask

###########################################################

testset.initialize(sess)
num_batches = testset.num_steps
masks    = [[] for _ in range(params.dimension+1)]
sam_mse  = [[] for _ in range(params.dimension+1)]
mean_mse = [[] for _ in range(params.dimension+1)]
nll      = [[] for _ in range(params.dimension+1)]

for n in range(num_batches):
    x, y = sess.run([testset.x, testset.y])
    b = np.zeros_like(x)
    mean_y, sam_y, logp_y = sess.run([model.mean_y, model.sam_y, model.logpy], {x_ph:x, y_ph:y, b_ph:b, m_ph:b})
    masks[0].append(b.copy())
    sam_mse[0].append(batch_mse(sam_y, y))
    mean_mse[0].append(batch_mse(mean_y, y))
    nll[0].append(-logp_y)
    for s in range(1, params.dimension+1):
        b = select_next(x, b)
        mean_y, sam_y, logp_y = sess.run([model.mean_y, model.sam_y, model.logpy], {x_ph:x, y_ph:y, b_ph:b, m_ph:b})
        masks[s].append(b.copy())
        sam_mse[s].append(batch_mse(sam_y, y))
        mean_mse[s].append(batch_mse(mean_y, y))
        nll[s].append(-logp_y)

masks = [np.concatenate(m, axis=0) for m in masks]
sam_mse  = [np.concatenate(e, axis=0) for e in sam_mse]
mean_mse = [np.concatenate(e, axis=0) for e in mean_mse]
nll = [np.concatenate(l, axis=0) for l in nll]

with gzip.open(f'{save_dir}/results.pgz', 'wb') as f:
    pickle.dump((masks, sam_mse, mean_mse, nll), f)

sam_rmse = [np.sqrt(np.mean(e)) for e in sam_mse]
mean_rmse = [np.sqrt(np.mean(e)) for e in mean_mse]
nll = [np.mean(l) for l in nll]
num = range(params.dimension+1)

fig = plt.figure()
plt.plot(num, sam_rmse, marker='x')
plt.xticks(num)
plt.xlabel('num feature')
plt.ylabel('RMSE')
plt.savefig(f'{save_dir}/sam_rmse.png')
plt.close('all')

fig = plt.figure()
plt.plot(num, mean_rmse, marker='x')
plt.xticks(num)
plt.xlabel('num feature')
plt.ylabel('RMSE')
plt.savefig(f'{save_dir}/mean_rmse.png')
plt.close('all')

fig = plt.figure()
plt.plot(num, nll, marker='x')
plt.xticks(num)
plt.xlabel('num feature')
plt.ylabel('NLL')
plt.savefig(f'{save_dir}/nll.png')
plt.close('all')

