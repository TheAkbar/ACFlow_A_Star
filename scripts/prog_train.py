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

# current observation
train_b = np.zeros([trainset.size, params.dimension], np.float32)
valid_b = np.zeros([validset.size, params.dimension], np.float32)
test_b = np.zeros([testset.size, params.dimension], np.float32)

def get_b(buffer, i):
    b = buffer[i]
    b = (b > 0).astype(np.float32)

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

def update_b(buffer, i, m, step):
    b = buffer[i]
    buffer[i] = b * (1-m) + m * (step+1)

def next_m(b):
    num = np.sum(b[0] == 0)
    for i in range(num):
        m = b.copy()
        for mi in m:
            d = np.where(mi == 0)[0]
            mi[d[i]] = 1
        yield m

##########################################################

def train_epoch():
    train_metrics = []
    num_steps = trainset.num_steps
    trainset.initialize(sess)
    for _ in range(num_steps):
        i, x, y = sess.run([trainset.i, trainset.x, trainset.y])
        b = get_b(train_b, i)
        m = gen_m(b)
        feed_dict = {x_ph:x, y_ph:y, b_ph:b, m_ph:m}
        metric, _ = sess.run([model.metric, model.train_op], feed_dict)
        train_metrics.append(metric)
    train_metrics = np.concatenate(train_metrics, axis=0)

    return np.mean(train_metrics)

def valid_epoch():
    valid_metrics = []
    num_steps = validset.num_steps
    validset.initialize(sess)
    for _ in range(num_steps):
        i, x, y = sess.run([validset.i, validset.x, validset.y])
        b = get_b(valid_b, i)
        m = gen_m(b)
        feed_dict = {x_ph:x, y_ph:y, b_ph:b, m_ph:m}
        metric = sess.run(model.metric, feed_dict)
        valid_metrics.append(metric)
    valid_metrics = np.concatenate(valid_metrics)

    return np.mean(valid_metrics)

def test_epoch():
    test_metrics = []
    num_steps = testset.num_steps
    testset.initialize(sess)
    for _ in range(num_steps):
        i, x, y = sess.run([testset.i, testset.x, testset.y])
        b = get_b(test_b, i)
        m = gen_m(b)
        feed_dict = {x_ph:x, y_ph:y, b_ph:b, m_ph:m}
        metric = sess.run(model.metric, feed_dict)
        test_metrics.append(metric)
    test_metrics = np.concatenate(test_metrics)

    return np.mean(test_metrics)

##########################################################

def expected_kl(p1, p2):
    '''
    compute KL(p1 || p2)
    p1: [B,N,C] probability
    p2: [B,N,C]
    due to an old scipy version, it only support sum over first dim.
    '''
    kl = entropy(np.transpose(p1, [2,0,1]), np.transpose(p2, [2,0,1]))

    return np.mean(kl, axis=1)

def cls_select_next(x, b):
    B = x.shape[0]
    N = params.num_samples
    d = params.dimension
    C = params.n_classes

    best_cmi = -np.ones([B]) * np.inf
    best_mask = np.zeros_like(b)
    # p(y | x_o): [B, C]
    pre_prob = sess.run(model.prob, {x_ph:x, b_ph:b, m_ph:b})
    pre_prob = np.repeat(np.expand_dims(pre_prob, axis=1), N, axis=1)
    # sample p(x_u | x_o)
    m = np.ones_like(b)
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

def reg_select_next(x, b):
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

def select_next(x, b):
    if 'classifier' in params.model:
        m = cls_select_next(x, b)
    elif 'regressor' in params.model:
        m = reg_select_next(x, b)
    else:
        raise Exception()

    return m - b

##########################################################

def train(step):
    if args.init:
        sess.run(initializer)

    best_train_metric = -np.inf
    best_valid_metric = -np.inf
    best_test_metric = -np.inf
    for epoch in range(params.epochs):
        train_metric = train_epoch()
        valid_metric = valid_epoch()
        test_metric = test_epoch()
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

def select(step):
    # train
    trainset.initialize(sess)
    num_steps = trainset.num_steps
    for _ in range(num_steps):
        i, x, y = sess.run([trainset.i, trainset.x, trainset.y])
        b = get_b(train_b, i)
        m = select_next(x, b)
        update_b(train_b, i, m, step)
    # valid
    validset.initialize(sess)
    num_steps = validset.num_steps
    for _ in range(num_steps):
        i, x, y = sess.run([validset.i, validset.x, validset.y])
        b = get_b(valid_b, i)
        m = select_next(x, b)
        update_b(valid_b, i, m, step)
    # test
    testset.initialize(sess)
    num_steps = testset.num_steps
    for _ in range(num_steps):
        i, x, y = sess.run([testset.i, testset.x, testset.y])
        b = get_b(test_b, i)
        m = select_next(x, b)
        update_b(test_b, i, m, step)

def save(step):
    save_dir = os.path.join(params.exp_dir, 'masks')
    os.makedirs(save_dir, exist_ok=True)
    with gzip.open(f'{save_dir}/step_{step}.pgz', 'wb') as f:
        data = {
            'train_b': train_b,
            'valid_b': valid_b,
            'test_b': test_b
        }
        pickle.dump(data, f)

def restore(step):
    if step < 0: return
    restore_from = os.path.join(params.exp_dir, f'weights/step_{step}.ckpt')
    saver.restore(sess, restore_from)

##########################################################
logging.info('starting training')
for step in range(params.dimension):
    logging.info('#'*10 + f'Step: {step}' + '#'*10)
    restore(step-1)
    train(step)
    restore(step)
    select(step)
    save(step)
