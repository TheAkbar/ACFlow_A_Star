import tensorflow as tf
import numpy as np

from .ACTAN import Flow

class Model(object):
    def __init__(self, hps):
        self.hps = hps

        self.flow = Flow(hps)

    def sample(self, x, b, m):
        B = tf.shape(x)[0]
        d = self.hps.dimension
        N = self.hps.num_samples
        x = tf.tile(tf.expand_dims(x,axis=1),[1,N,1])
        x = tf.reshape(x, [B*N,d])
        b = tf.tile(tf.expand_dims(b,axis=1),[1,N,1])
        b = tf.reshape(b, [B*N,d])
        m = tf.tile(tf.expand_dims(m,axis=1),[1,N,1])
        m = tf.reshape(m, [B*N,d])

        sam = self.flow.inverse(x, b, m)
        sam = tf.reshape(sam, [B,N,d])

        return sam 

    def build(self, x, y, b, m):
        self.log_likel = self.flow.forward(x, b, m)
        self.x_sam = self.sample(x, b, m)

        # loss
        loss = tf.reduce_mean(-self.log_likel)
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
        self.metric = self.log_likel
