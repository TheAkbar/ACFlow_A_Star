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

############################################################

# mask_dir = os.path.join(params.exp_dir, 'masks')
# with gzip.open(f'{mask_dir}/step_{params.dimension-1}.pgz', 'rb') as f:
#     data = pickle.load(f)
# test_b = data['test_b']

############################################################

def get_b(i, s):
    mask_dir = os.path.join(params.exp_dir, 'masks')
    with gzip.open(f'{mask_dir}/step_{s}.pgz', 'rb') as f:
        data = pickle.load(f)
    test_b = data['test_b']

    b = test_b[i].copy()
    b = (b == s+1).astype(np.float32)

    return b

def restore(step):
    if step < 0: return
    restore_from = os.path.join(params.exp_dir, f'weights/step_{step}.ckpt')
    saver.restore(sess, restore_from)

def batch_mse(s, y):
    '''
    s: [B,N]
    y: [B]
    '''
    mse = (s - y[:,None]) ** 2
    mse = np.mean(mse, axis=1)

    return mse

############################################################

num_batches = testset.num_steps
masks = [[] for _ in range(params.dimension)]
mse   = [[] for _ in range(params.dimension)]
nll   = [[] for _ in range(params.dimension)]
for step in range(params.dimension):
    restore(step)
    testset.initialize(sess)
    for n in range(num_batches):
        i, x, y = sess.run([testset.i, testset.x, testset.y])
        b = get_b(i, step)
        sam_y, logp_y = sess.run([model.sam_y, model.logpy], {x_ph:x, y_ph:y, b_ph:b, m_ph:b})
        masks[step].append(b.copy())
        mse[step].append(batch_mse(sam_y, y))
        nll[step].append(-logp_y)

masks = [np.concatenate(m, axis=0) for m in masks]
mse  = [np.concatenate(e, axis=0) for e in mse]
nll = [np.concatenate(l, axis=0) for l in nll]

with gzip.open(f'{save_dir}/results.pgz', 'wb') as f:
    pickle.dump((masks, mse, nll), f)

rmse = [np.sqrt(np.mean(e)) for e in mse]
nll = [np.mean(l) for l in nll]
num = range(1, params.dimension+1)

fig = plt.figure()
plt.plot(num, rmse, marker='x')
plt.xticks(num)
plt.xlabel('num feature')
plt.ylabel('RMSE')
plt.savefig(f'{save_dir}/rmse.png')
plt.close('all')

fig = plt.figure()
plt.plot(num, nll, marker='x')
plt.xticks(num)
plt.xlabel('num feature')
plt.ylabel('NLL')
plt.savefig(f'{save_dir}/nll.png')
plt.close('all')