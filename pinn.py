import tensorflow as tf
import numpy as np
import time
from scipy import optimize

class CircuitPINN:
    # Initialize the class
    def __init__(self, t0, v0, i0, R, L, layers):

        # Data for MSEf
        self.t0_f = t0[50:]
        self.v0_f = v0[50:]

        # Data for MSEu
        self.t0 = t0[:50]
        self.i0 = i0[:50]

        self.R = R  # Resistance
        self.L = L  # Indutance

        # Initialize NNs
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        # tf Placeholders
        self.t0_tf = tf.constant([self.t0], dtype=tf.float32)

        self.i0_tf = tf.constant([self.i0], dtype=tf.float32)

        self.t0_f_tf = tf.constant([self.t0_f], dtype=tf.float32)
        self.v0_f_tf = tf.constant([self.v0_f], dtype=tf.float32)

        # tf Graphs
        i_pred_list = []
        f_i_pred_list = []

        for j in range(self.t0_tf.shape[1]):
            item_i_pred = self.net_i(tf.slice(self.t0_tf, [0, j], [1, 1]))
            i_pred_list.append(item_i_pred[0][0].numpy())
        for j in range(self.t0_f_tf.shape[1]):
            item_f_i_pred = self.net_f_i(tf.slice(self.t0_f_tf, [0, j], [1, 1]), tf.slice(self.v0_f_tf, [0, j], [1, 1]))
            f_i_pred_list.append(item_f_i_pred[0][0].numpy())

        self.i_pred = tf.constant([i_pred_list])
        self.f_i_pred = tf.constant([f_i_pred_list])

        # Loss
        self.loss = tf.reduce_mean(tf.square(self.i0_tf - self.i_pred)) + \
                    tf.reduce_mean(tf.square(self.f_i_pred))

        # Optimizers
        # self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
        #                                                         method='L-BFGS-B',
        #                                                         options={'maxiter': 50000,
        #                                                                  'maxfun': 50000,
        #                                                                  'maxcor': 50,
        #                                                                  'maxls': 50,
        #                                                                  'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]]), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, t, weights, biases):
        num_layers = len(weights) + 1
        U = t
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            U = tf.tanh(tf.add(tf.matmul(U, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(U, W), b)
        return Y

    def net_i(self, t):
        return self.neural_net(t, self.weights, self.biases)

    def net_f_i(self, t, v):
        with tf.GradientTape() as gt:
            gt.watch(t)
            i = self.net_i(t)
        di_dt = gt.gradient(i, t)

        return di_dt + (self.R / self.L) * i - (1 / self.L) * v

    def callback(self, loss):
        print('Loss:', loss)

    def train(self, nIter):

        tf_dict = {self.x0_tf: self.x0, self.t0_tf: self.t0,
                   self.u0_tf: self.u0, self.v0_tf: self.v0,
                   self.x_lb_tf: self.x_lb, self.t_lb_tf: self.t_lb,
                   self.x_ub_tf: self.x_ub, self.t_ub_tf: self.t_ub,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value, elapsed))
                start_time = time.time()

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)

    def predict(self, X_star):

        tf_dict = {self.x0_tf: X_star[:, 0:1], self.t0_tf: X_star[:, 1:2]}

        u_star = self.sess.run(self.u0_pred, tf_dict)
        v_star = self.sess.run(self.v0_pred, tf_dict)

        tf_dict = {self.x_f_tf: X_star[:, 0:1], self.t_f_tf: X_star[:, 1:2]}

        f_u_star = self.sess.run(self.f_u_pred, tf_dict)
        f_v_star = self.sess.run(self.f_v_pred, tf_dict)

        return u_star, v_star, f_u_star, f_v_star
