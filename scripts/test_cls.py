import os
import sys
p = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(p)
import argparse
import logging
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
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

def expected_kl(p1, p2):
    '''
    compute KL(p1 || p2)
    p1: [B,N,C] probability
    p2: [B,N,C]
    due to an old scipy version, it only support sum over first dim.
    '''
    kl = entropy(np.transpose(p1, [2,0,1]), np.transpose(p2, [2,0,1]))

    return np.mean(kl, axis=1)

def next_m(b):
    num = np.sum(b[0] == 0)
    for i in range(num):
        m = b.copy()
        for mi in m:
            d = np.where(mi == 0)[0]
            mi[d[i]] = 1
        yield m

def select_next(x, b, pre_prob):
    B = x.shape[0]
    N = params.num_samples
    d = params.dimension
    C = params.n_classes

    best_cmi = -np.ones([B]) * np.inf
    best_mask = np.zeros_like(b)
    pre_prob = np.repeat(np.expand_dims(pre_prob, axis=1), N, axis=1)
    m = np.ones_like(x)
    sam_all = sess.run(model.sam, {x_ph:x, b_ph:b, m_ph:m})
    for m in next_m(b):
        q = np.expand_dims(m*(1-b), axis=1)
        sam = sam_all.copy() * q + np.expand_dims(x, axis=1) * (1-q)
        sam = sam.reshape([B*N, d])
        b_tile = np.repeat(np.expand_dims(b, axis=1), N, axis=1).reshape([B*N, d])
        m_tile = np.repeat(np.expand_dims(m, axis=1), N, axis=1).reshape([B*N, d])
        post_prob = sess.run(model.prob, {x_ph:sam, b_ph:b_tile, m_ph:m_tile})
        post_prob = post_prob.reshape([B,N,C])
        cmi = expected_kl(post_prob, pre_prob)
        Bid = cmi > best_cmi
        best_mask[Bid] = m[Bid].copy()
        best_cmi[Bid] = cmi[Bid].copy()

    return best_mask

###########################################################

testset.initialize(sess)
num_batches = testset.num_steps
masks = [[] for _ in range(params.dimension+1)]
probs = [[] for _ in range(params.dimension+1)]
acc   = [[] for _ in range(params.dimension+1)]
for n in range(num_batches):
    x, y = sess.run([testset.x, testset.y])
    b = np.zeros_like(x)
    prob = sess.run(model.prob, {x_ph:x, b_ph:b, m_ph:b})
    masks[0].append(b.copy())
    probs[0].append(prob.copy())
    acc[0].append(np.argmax(prob, axis=1) == y)
    for s in range(1, params.dimension+1):
        b = select_next(x, b, prob)
        prob = sess.run(model.prob, {x_ph:x, b_ph:b, m_ph:b})
        masks[s].append(b.copy())
        probs[s].append(prob.copy())
        acc[s].append(np.argmax(prob, axis=1) == y)

masks = [np.concatenate(m, axis=0) for m in masks]
probs = [np.concatenate(p, axis=0) for p in probs]
acc   = [np.concatenate(a, axis=0) for a in acc]

with gzip.open(f'{save_dir}/results.pgz', 'wb') as f:
    pickle.dump((masks, probs, acc), f)

acc = [np.mean(a) for a in acc]
num = range(params.dimension+1)
fig = plt.figure()
plt.plot(num, acc, marker='x')
plt.xticks(num)
plt.xlabel('num feature')
plt.ylabel('accuracy')
plt.savefig(f'{save_dir}/acc.png')
plt.close('all')