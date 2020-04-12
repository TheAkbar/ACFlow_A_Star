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
import copy

from datasets import get_dataset
from models import get_model
from utils.hparams import HParams

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str)
parser.add_argument('--init', action='store_true')
args = parser.parse_args()
params = HParams(args.cfg_file)
pprint(params.dict)

os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu
np.random.seed(params.seed)
tf.set_random_seed(params.seed)

############################################################
logging.basicConfig(filename=params.exp_dir + '/train.log',
                    filemode='w',
                    level=logging.INFO,
                    format='%(message)s')
logging.info(pformat(params.dict))
############################################################

trainset = get_dataset('train', params)
validset = get_dataset('valid', params)
testset = get_dataset('test', params)
logging.info(f"trainset: {trainset.size} \
               validset: {validset.size} \
               testset: {testset.size}")

x_ph = tf.placeholder(tf.float32, [None, params.dimension])
y_ph = tf.placeholder(tf.float32, [None])
b_ph = tf.placeholder(tf.float32, [None, params.dimension])
m_ph = tf.placeholder(tf.float32, [None, params.dimension])

model = get_model(params)
model.build(x_ph, y_ph, b_ph, m_ph)

total_params = 0
trainable_variables = tf.trainable_variables()
logging.info('=' * 20)
logging.info("Variables:")
logging.info(pformat(trainable_variables))
for k, v in enumerate(trainable_variables):
    num_params = np.prod(v.get_shape().as_list())
    total_params += num_params

logging.info("TOTAL TENSORS: %d TOTAL PARAMS: %f[M]" % (
    k + 1, total_params / 1e6))
logging.info('=' * 20)

##########################################################
initializer = tf.global_variables_initializer()
saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

config = tf.ConfigProto()
config.log_device_placement = True
config.allow_soft_placement = True
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

sess.run(initializer)

##########################################################

def get_b(x, s):
    B, d = x.shape
    b = np.zeros_like(x)
    for i in range(B):
        w = np.random.choice(d, [s], replace=False)
        b[i,w] = 1.

    return b

def gen_m(b):
    m = copy.deepcopy(b)
    for mi in m:
        w = list(np.where(mi == 0)[0])
        w.append(-1)
        w = np.random.choice(w)
        if w >= 0:
            mi[w] = 1.

    return m

##########################################################

def train_epoch(step):
    train_metrics = []
    num_steps = trainset.num_steps
    trainset.initialize(sess)
    for _ in range(num_steps):
        x, y = sess.run([trainset.x, trainset.y])
        b = get_b(x, step)
        m = gen_m(b)
        feed_dict = {x_ph:x, y_ph:y, b_ph:b, m_ph:m}
        metric, _ = sess.run([model.metric, model.train_op], feed_dict)
        train_metrics.append(metric)
    train_metrics = np.concatenate(train_metrics, axis=0)

    return np.mean(train_metrics)

def valid_epoch(step):
    valid_metrics = []
    num_steps = validset.num_steps
    validset.initialize(sess)
    for _ in range(num_steps):
        x, y = sess.run([validset.x, validset.y])
        b = get_b(x, step)
        m = gen_m(b)
        feed_dict = {x_ph:x, y_ph:y, b_ph:b, m_ph:m}
        metric = sess.run(model.metric, feed_dict)
        valid_metrics.append(metric)
    valid_metrics = np.concatenate(valid_metrics)

    return np.mean(valid_metrics)

def test_epoch(step):
    test_metrics = []
    num_steps = testset.num_steps
    testset.initialize(sess)
    for _ in range(num_steps):
        x, y = sess.run([testset.x, testset.y])
        b = get_b(x, step)
        m = gen_m(b)
        feed_dict = {x_ph:x, y_ph:y, b_ph:b, m_ph:m}
        metric = sess.run(model.metric, feed_dict)
        test_metrics.append(metric)
    test_metrics = np.concatenate(test_metrics)

    return np.mean(test_metrics)

##########################################################

def train(step):
    if args.init:
        sess.run(initializer)

    best_train_metric = -np.inf
    best_valid_metric = -np.inf
    best_test_metric = -np.inf
    for epoch in range(params.epochs):
        train_metric = train_epoch(step)
        valid_metric = valid_epoch(step)
        test_metric = test_epoch(step)
        # save
        if train_metric > best_train_metric:
            best_train_metric = train_metric
        if valid_metric > best_valid_metric:
            best_valid_metric = valid_metric
        if test_metric > best_test_metric:
            best_test_metric = test_metric

        logging.info("Epoch %d, train: %.4f/%.4f, valid: %.4f/%.4f test: %.4f/%.4f" %
                    (epoch, train_metric, best_train_metric, 
                    valid_metric, best_valid_metric,
                    test_metric, best_test_metric))
        sys.stdout.flush()

    save_path = os.path.join(params.exp_dir, f'weights/step_{step}.ckpt')
    saver.save(sess, save_path)

def restore(step):
    if step < 0: return
    restore_from = os.path.join(params.exp_dir, f'weights/step_{step}.ckpt')
    saver.restore(sess, restore_from)

##########################################################
logging.info('starting training')
for step in range(params.dimension+1):
    logging.info('#'*10 + f'Step: {step}' + '#'*10)
    restore(step-1)
    train(step)
