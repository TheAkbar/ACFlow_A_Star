import tensorflow as tf
import numpy as np

from .ACTAN import Flow

class Model(object):
    def __init__(self, hps):
        self.hps = hps

        self.flow = Flow(hps)

    def classify(self, x, b, m):
        B = tf.shape(x)[0]
        d = self.hps.dimension
        N = self.hps.n_classes

        x = tf.tile(tf.expand_dims(x,axis=1), [1,N,1])
        x = tf.reshape(x, [B*N,d])
        b = tf.tile(tf.expand_dims(b,axis=1), [1,N,1])
        b = tf.reshape(b, [B*N,d])
        m = tf.tile(tf.expand_dims(m,axis=1), [1,N,1])
        m = tf.reshape(m, [B*N,d])
        y = tf.tile(tf.expand_dims(tf.range(N),axis=0), [B,1])
        y = tf.reshape(y, [B*N])

        # log p(x_u | x_o, y)
        logp = self.flow.cond_forward(x, y, b, m)
        # logits
        logits = tf.reshape(logp, [B,N])

        return logits

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
        y = tf.random_uniform([B*N], dtype=tf.int64, minval=0, maxval=self.hps.n_classes)

        sam = self.flow.cond_inverse(x, y, b, m)
        sam = tf.reshape(sam, [B,N,d])

        return sam

    def build(self, x, y, b, m):
        y = tf.cast(y, tf.int64)
        # log p(x_u | x_o, y)
        self.logpu = self.classify(x, b, m)
        # log p(x_o | y)
        self.logpo = self.classify(x, b*(1-b), b)
        # log p(x_u, x_o | y)
        self.logpuo = self.classify(x, m*(1-m), m)

        # logits: log p (x_u, x_u | y)
        if self.hps.version == 'v1':
            self.logits = self.logpu + self.logpo 
        elif self.hps.version == 'v2':
            self.logits = self.logpuo
        else:
            raise Exception()

        # p(y | x_u, x_o)
        self.prob = tf.nn.softmax(self.logits)
        self.pred = tf.argmax(self.logits, axis=1)
        self.acc = tf.cast(tf.equal(self.pred, y), tf.float32)

        # log p(x_u | x_o)
        self.log_likel = (tf.reduce_logsumexp(self.logpu + self.logpo, axis=1) - 
                          tf.reduce_logsumexp(self.logpo, axis=1))
        
        # sample p(x_u | x_o)
        self.sam = self.sample(x, b, m)

        # loss
        xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=y)
        xent = tf.reduce_mean(xent)
        tf.summary.scalar('xent', xent)
        nll = tf.reduce_mean(-self.log_likel)
        tf.summary.scalar('nll', nll)
        
        if self.hps.loss == 'xent':
            loss = xent
        elif self.hps.loss == 'sum':
            loss = xent + self.hps.lambda_nll * nll
        elif self.hps.loss == 'logsumexp':
            loss = -tf.reduce_logsumexp(tf.stack([-xent, -nll]), axis=0)
        else:
            raise Exception()
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

        # metric
        self.metric = self.acc