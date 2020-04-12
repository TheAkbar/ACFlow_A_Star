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
config.log_device_placement = True
config.allow_soft_placement = True
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

saver = tf.train.Saver(tf.global_variables())
weights_dir = os.path.join(params.exp_dir, 'weights')
logging.info(f'Restoring parameters from {weights_dir}')
restore_from = tf.train.latest_checkpoint(weights_dir)
saver.restore(sess, restore_from)

print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

###########################################################4

# testset.initialize(sess)
# num_batches = testset.num_steps
# test_ll = []
# for n in range(num_batches):
#     x, y, b = sess.run([testset.x, testset.y, testset.b])
#     m = np.ones_like(b)
#     ll = sess.run(model.log_likel, {x_ph:x, b_ph:b, m_ph:m})
#     test_ll.append(ll)
# test_ll = np.concatenate(test_ll, axis=0)
# test_ll = np.mean(test_ll)
# logging.info(f'test log likelihood: {test_ll}')
