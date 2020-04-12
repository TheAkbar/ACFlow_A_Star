import numpy as np
import heapq
import copy
import os
import sys
p = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(p)
import argparse
import logging
import tensorflow as tf
import pickle
from pprint import pformat, pprint

from datasets import get_dataset
from models import get_model
from utils.hparams import HParams

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str)
parser.add_argument('--lambda_c', type=str)
args = parser.parse_args()


class Node:
    def __init__(self, vertex, vertices_order, edge_path, adj_m, beta, h, g):
        self.vertex = vertex
        self.vertices_order = vertices_order
        self.edge_path = edge_path
        self.adj_matrix = adj_m
        self.beta = beta
        self.h = h
        self.g = g
        self.f = h + g

    def __hash__(self):
        return repr(self).__hash__()

    def __repr__(self):
        return '#'.join([str(x) for x in sorted(self.vertices_order)])

    def __lt__(self, other):
        return self.f < other.f


def list_to_key(vertex_path):
    path = [str(x) for x in sorted(vertex_path)]
    return '#'.join(path)


def run_lasso(data, parent_indices, child_indices, lambda_c, run_model):
    m = np.zeros(data.shape)
    m[:, parent_indices] = 1
    m[:, child_indices] = 1

    b = np.ones(data.shape)
    b[:, child_indices] = 0

    curr_g = np.mean(run_model(data, b, m))
    # curr_g = y_diff.dot(y_diff) + lambda_c * np.sum(np.abs(beta))
    return 1.0, curr_g


def remove(all_v, parent):
    return [x for x in all_v if x not in parent]


def get_h_score(data, lambda_c, run_model):
    pre_computed_lasso_scores = []
    _, q = np.shape(data)
    sum_score = 0
    for j in range(q):
        # print('pre-computing score for vertex_{}'.format(j))
        mask = np.ones(q, dtype=np.bool)
        mask[j] = 0
        # y = data[:, j]
        # x = data[:, mask]
        beta, score = run_lasso(data, mask, j, lambda_c, run_model)
        # y_diff = y - x.dot(beta)
        # score = y_diff.dot(y_diff) + lambda_c * np.sum(np.abs(beta))
        pre_computed_lasso_scores.append(score)
        sum_score += score

    return sum_score, pre_computed_lasso_scores


def a_star_lasso(data, lambda_c, run_model):
    n, q = np.shape(data)

    all_vertices = np.arange(q)
    level = 0

    min_heap = []
    visited = set()

    print('running full A* lasso')
    sum_score, pc_lasso_score = get_h_score(data, lambda_c, run_model)
    print(sum_score)
    start_node = Node(-1, list(), list(), np.zeros((q, q)), np.zeros((q, q)), sum_score, 0)
    heapq.heappush(min_heap, start_node)
    # visited.add(repr(start_node))

    while len(min_heap) > 0:
        cur_node = heapq.heappop(min_heap)
        visited.add(repr(cur_node))

        if len(cur_node.vertices_order) == q:
            return cur_node, level

        candidate_child = remove(all_vertices, cur_node.vertices_order)

        for child in candidate_child:
            vertices_order = copy.deepcopy(cur_node.vertices_order)
            vertices_order.append(child)
            if list_to_key(vertices_order) in visited:
                continue

            edge_path = copy.deepcopy(cur_node.edge_path)
            adj_m = copy.deepcopy(cur_node.adj_matrix)
            beta = copy.deepcopy(cur_node.beta)
            h_score = cur_node.h
            g_score = cur_node.g

            y = data[:, child]

            if len(cur_node.vertices_order) == 0:
                curr_g = y.dot(y)
            else:
                parent = cur_node.vertices_order
                # parent_data = data[:, parent]

                beta_child, curr_g = run_lasso(data, parent, child, lambda_c, run_model)
                beta[parent, child] = beta_child
                identity = beta_child != 0
                adj_m[parent, child] = identity

                # y_diff = y - parent_data.dot(beta_child)
                # curr_g = y_diff.dot(y_diff) + lambda_c * np.sum(np.abs(beta_child))
            g_score += curr_g
            h_score -= pc_lasso_score[child]

            child_node = Node(child, vertices_order, edge_path, adj_m, beta, h_score, g_score)
            heapq.heappush(min_heap, child_node)
            min_heap = min_heap[:200]
            # heapq.heapify(min_heap)

if __name__ == '__main__':
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

    with open(params.dfile, 'rb') as f:
        data = pickle.load(f)['test']

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
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver(tf.global_variables())
        weights_dir = os.path.join(params.exp_dir, 'weights')
        logging.info(f'Restoring parameters from {weights_dir}')
        restore_from = tf.train.latest_checkpoint(weights_dir)
        saver.restore(sess, restore_from)

        run_model = lambda x, b, m: sess.run(
            model.log_likel, {
                x_ph: x, b_ph: b, m_ph: m
            }
        )

        node, _ = a_star_lasso(data, args.lambda_c, run_model)
        print(node.adj_matrix)
        print(node.vertices_order)
        print('completed')