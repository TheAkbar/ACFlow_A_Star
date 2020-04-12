import os
import sys
p = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(p)
import argparse
import logging
import tensorflow as tf
import numpy as np
from pprint import pformat, pprint

from datasets import get_dataset
from models import get_model
from utils.hparams import HParams

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str)
parser.add_argument('--lambda_1norm', type=float)
parser.add_argument('--lambda_2norm', type=float)
args = parser.parse_args()
params = HParams(args.cfg_file)
params.lambda_norm = args.lambda_1norm
params.lambda_2norm = args.lambda_2norm

os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu
np.random.seed(params.seed)
tf.set_random_seed(params.seed)

############################################################
logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format='%(message)s')
logging.info(pformat(params.dict))
writer = tf.summary.FileWriter(params.exp_dir + '/summaries')
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

##########################################################

def train():
    train_metrics = []
    train_losses = []
    num_steps = trainset.num_steps
    trainset.initialize(sess)
    norms = []
    for i in range(num_steps):
        x, y, b, m = sess.run([trainset.x, trainset.y, trainset.b, trainset.m])
        feed_dict = {x_ph:x, y_ph:y, b_ph:b, m_ph:m}
        graph = tf.get_default_graph()
        sparse_mat = graph.get_tensor_by_name('sparsity_mat:0').eval(session=sess)
        # print(sparse_mat)
        norm, metric, _, _, loss, norm_val = sess.run(
            [model.norm, model.metric, model.train_op, model.mask_op, model.loss, model.norm_val], feed_dict
        )
        # print(f'norm_val: {norm_val}')
        norms.append(norm_val)
        train_metrics.append(metric)
        train_losses.append(loss)
        if (params.print_freq > 0) and (i % params.print_freq == 0):
            metric = np.mean(metric)
            logging.info('step: %d metric:%.4f ' % (i, metric))
    sess.run(model.norm_update)
    train_metrics = np.concatenate(train_metrics, axis=0)
    train_losses = np.array(train_losses)

    print(norm)
    return np.mean(train_metrics), np.mean(train_losses)

def valid():
    valid_metrics = []
    num_steps = validset.num_steps
    validset.initialize(sess)
    for i in range(num_steps):
        x, y, b, m = sess.run([validset.x, validset.y, validset.b, validset.m])
        feed_dict = {x_ph:x, y_ph:y, b_ph:b, m_ph:m}
        metric = sess.run(model.metric, feed_dict)
        valid_metrics.append(metric)
    valid_metrics = np.concatenate(valid_metrics)

    return np.mean(valid_metrics)

def test():
    test_metrics = []
    num_steps = testset.num_steps
    testset.initialize(sess)
    for i in range(num_steps):
        x, y, b, m = sess.run([testset.x, testset.y, testset.b, testset.m])
        feed_dict = {x_ph:x, y_ph:y, b_ph:b, m_ph:m}
        metric = sess.run(model.metric, feed_dict)
        test_metrics.append(metric)
    test_metrics = np.concatenate(test_metrics)

    return np.mean(test_metrics)

##########################################################
np.set_printoptions(linewidth=200)
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

initializer = tf.global_variables_initializer()
saver = tf.train.Saver(tf.global_variables())

config = tf.ConfigProto()
config.log_device_placement = False
config.allow_soft_placement = True
config.gpu_options.allow_growth = True
graph = tf.get_default_graph()
with tf.Session(config=config) as sess:

    sess.run(initializer)

    logging.info('starting training')
    best_train_metric = -np.inf
    best_valid_metric = -np.inf
    best_test_metric = -np.inf
    for epoch in range(params.epochs):
        train_metric, train_loss = train()
        valid_metric = valid()
        test_metric = test()
        # save
        if train_metric > best_train_metric:
            best_train_metric = train_metric
        if valid_metric > best_valid_metric:
            best_valid_metric = valid_metric
            save_path = os.path.join(params.exp_dir, 'weights/params.ckpt')
            saver.save(sess, save_path)
        if test_metric > best_test_metric:
            best_test_metric = test_metric

        logging.info("Epoch %d, loss: %.3f, train: %.3f/%.3f, valid: %.3f/%.3f test: %.3f/%.3f" %
                    (epoch, train_loss, train_metric, best_train_metric, 
                    valid_metric, best_valid_metric,
                    test_metric, best_test_metric))
        sparse_mat = graph.get_tensor_by_name('sparsity_mat:0').eval(session=sess)
        print(np.triu(sparse_mat) + np.triu(sparse_mat).T)
        sys.stdout.flush()