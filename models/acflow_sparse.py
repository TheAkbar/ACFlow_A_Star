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
        # make the beta mat symmetric and subtract out the diagonal
        beta_mat = tf.get_variable(
            'sparsity_mat', shape=(self.hps.dimension, self.hps.dimension),
            initializer=tf.zeros_initializer, trainable=False
        )
        sym_beta_mat = tf.matrix_band_part(beta_mat, 0, -1) - tf.matrix_band_part(beta_mat, 0, 0)
        sym_beta_mat += tf.transpose(sym_beta_mat)

        # 1 - b indexes the appropriate row of beta_mat and element wise product with b ensures
        # that only the conditioning information is affected
        full_mask = m * b
        x = x * tf.matmul(1 - b, sym_beta_mat) * full_mask + x * (1 - full_mask)
        self.log_likel = self.flow.forward(x, b, m)
        self.x_sam = self.sample(x, b, m)

        print(f'lambda 1 norm: {self.hps.lambda_norm}')

        # l1_reg = self.hps.lambda_norm * tf.reduce_sum(tf.abs(sym_beta_mat))
        # l1_reg = self.hps.lambda_norm * tf.nn.l2_loss(sym_beta_mat)
        self.norm_val = tf.reduce_sum(tf.abs(sym_beta_mat))
        loss = tf.reduce_mean(-self.log_likel)
        self.loss = loss
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

        self.norm = tf.Variable(self.hps.min_norm, trainable=False)
        self.norm_update = tf.assign(self.norm, tf.minimum(
            self.hps.norm_up * self.norm, self.hps.max_norm
        ))
        # update the beta matrix
        mask_grad = optimizer.compute_gradients(self.loss, [beta_mat])[0][0]
        self.learning_rate = mask_grad
        d_mask = beta_mat - 100*learning_rate*mask_grad
        new_mask = tf.sign(d_mask)*tf.maximum(0.0, tf.abs(d_mask)-self.norm)
        self.mask_op = tf.assign(beta_mat, new_mask)

        # summary
        self.summ_op = tf.summary.merge_all()

        # metric
        self.metric = self.log_likel
