import tensorflow as tf
import numpy as np
import copy

from .ACTAN import Flow

class Model(object):
    def __init__(self, hps):
        self.hps = copy.deepcopy(hps)
        # concate x and y
        self.hps.dimension += 1

        self.flow = Flow(self.hps)

    def sample(self, x, b, m):
        B = tf.shape(x)[0]
        d = self.hps.dimension
        N = self.hps.num_samples

        x = tf.tile(tf.expand_dims(x,axis=1), [1,N,1])
        x = tf.reshape(x, [B*N,d])
        b = tf.tile(tf.expand_dims(b,axis=1), [1,N,1])
        b = tf.reshape(b, [B*N,d])
        m = tf.tile(tf.expand_dims(m,axis=1), [1,N,1])
        m = tf.reshape(m, [B*N,d])

        sam = self.flow.inverse(x, b, m)
        sam = tf.reshape(sam, [B,N,d])

        return sam

    def build(self, x, y, b, m):
        B = tf.shape(x)[0]
        d = self.hps.dimension

        xy = tf.concat([x, tf.expand_dims(y,axis=1)], axis=1)
        # log p(y | x_u, x_o)
        by = tf.concat([m, tf.zeros((B,1),dtype=tf.float32)], axis=1)
        my = tf.concat([m, tf.ones((B,1),dtype=tf.float32)], axis=1)
        self.logpy = self.flow.forward(xy, by, my)

        # sample p(y | x_u, x_o)
        sam_y = self.sample(xy, by, my)
        self.sam_y = sam_y[:,:,-1]

        # mean of p(y | x_u, x_o)
        mean_y = self.flow.mean(xy, by, my)
        self.mean_y = mean_y[:,-1]

        # log p(x_u | x_o)
        bu = tf.concat([b, tf.zeros((B,1),dtype=tf.float32)], axis=1)
        mu = tf.concat([m, tf.zeros((B,1),dtype=tf.float32)], axis=1)
        self.logpu = self.flow.forward(xy, bu, mu)

        # sample p(x_u | x_o)
        sam_u = self.sample(xy, bu, mu)
        self.sam_u = sam_u[:,:,:-1]

        # log p(x_u, y | x_o)
        bj = tf.concat([b, tf.zeros((B,1),dtype=tf.float32)], axis=1)
        mj = tf.concat([m, tf.ones((B,1),dtype=tf.float32)], axis=1)
        self.logpj = self.flow.forward(xy, bj, mj)

        # sample p(x_u, y | x_o)
        self.sam_j = self.sample(xy, bj, mj)

        # mean of p(x_u, y | x_o)
        self.mean_j = self.flow.mean(xy, bj, mj)

        # log p(y | x_o)
        bo = tf.concat([b, tf.zeros((B,1), dtype=tf.float32)], axis=1)
        mo = tf.concat([b, tf.ones((B,1), dtype=tf.float32)], axis=1)
        self.logpo = self.flow.forward(xy, bo, mo)

        # log ratio
        self.log_ratio = self.logpy - self.logpo

        # loss
        if self.hps.loss == 'v1':
            loss = -tf.reduce_mean(self.logpy) - tf.reduce_mean(self.logpj)
            self.metric = self.logpy + self.logpj
        elif self.hps.loss == 'v2':
            loss = -tf.reduce_mean(self.logpy) - tf.reduce_mean(self.logpj) - tf.reduce_mean(self.logpu)
            self.metric = self.logpy + self.logpj + self.logpu
        elif self.hps.loss == 'imp':
            bc = tf.cast(tf.random_uniform([B,d]) > 0.5, tf.float32)
            mc = tf.cast(tf.random_uniform([B,d]) > 0.5, tf.float32)
            bc = mc * bc
            logp = self.flow.forward(xy, bc, mc)
            loss = -tf.reduce_mean(logp)
            self.metric = logp
        else:
            raise Exception()

        if self.hps.lambda_mse > 0:
            mse = tf.reduce_mean(tf.reduce_sum(tf.square(self.mean_j - xy), axis=1))
            loss += self.hps.lambda_mse * mse

        tf.summary.scalar('loss', loss)

        # train
        self.global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.inverse_time_decay(
            self.hps.lr, self.global_step,
            self.hps.decay_steps, self.hps.decay_rate,
            staircase=True)
        tf.summary.scalar('lr', learning_rate)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=0.9, beta2=0.999, epsilon=1e-08,
            use_locking=False, name="Adam")
        grads_and_vars = optimizer.compute_gradients(
            loss, tf.trainable_variables())
        grads, vars_ = zip(*grads_and_vars)
        if self.hps.clip_gradient > 0:
            grads, gradient_norm = tf.clip_by_global_norm(
                grads, clip_norm=self.hps.clip_gradient)
            gradient_norm = tf.check_numerics(
                gradient_norm, "Gradient norm is NaN or Inf.")
            tf.summary.scalar('gradient_norm', gradient_norm)
        capped_grads_and_vars = zip(grads, vars_)
        self.train_op = optimizer.apply_gradients(
            capped_grads_and_vars, global_step=self.global_step)

        # summary
        self.summ_op = tf.summary.merge_all()



        