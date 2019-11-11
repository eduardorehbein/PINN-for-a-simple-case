import tensorflow as tf
import numpy as np
import time
from scipy import optimize

# TODO: Improve it basing on https://github.com/pierremtb/PINNs-TF2.0/blob/master/utils/neuralnetwork.py

class CircuitPINN:

    def __init__(self, R, L, hidden_layers, learning_rate):
        # Circuit parameters
        self.R = R  # Resistance
        self.L = L  # Inductance

        # Initialize NN
        self.layers = [1] + hidden_layers + [1]
        self.weights, self.biases = self.initialize_NN(self.layers)

        # Optimizer
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(1, num_layers):
            W = self.xavier_init(size=[layers[l], layers[l - 1]])
            b = tf.Variable(tf.zeros([layers[l], 1]), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def predict(self, numpy_or_list_t):
        t = tf.constant(numpy_or_list_t, dtype=tf.float32)
        return self.i(t)

    def i(self, t):
        num_layers = len(self.weights) + 1
        U = t
        for l in range(0, num_layers - 2):
            W = self.weights[l]
            b = self.biases[l]
            U = tf.tanh(tf.add(tf.matmul(W, U), b))
        W = self.weights[-1]
        b = self.biases[-1]
        Y = tf.add(tf.matmul(W, U), b)
        return Y

    def f(self, t, v):
        # ODE: Ldi_dt = v - Ri
        with tf.GradientTape() as gtf:
            gtf.watch(t)
            i = self.i(t)
        di_dt = gtf.gradient(i, t)

        return di_dt + (self.R / self.L) * i - (1 / self.L) * v

    def train(self, train_t, train_i, train_v, epochs=1, train_f_percent=0.99):
        # Data for f loss
        train_f_index = int(train_f_percent * train_t.shape[1])
        f_t_train = tf.constant(train_t[:, :train_f_index], dtype=tf.float32)
        f_v_train = tf.constant(train_v[:, :train_f_index], dtype=tf.float32)

        # Data for u loss
        u_t_train = tf.constant(train_t[:, train_f_index:], dtype=tf.float32)
        u_i_train = tf.constant(train_i[:, train_f_index:], dtype=tf.float32)

        for j in range(epochs):
            grad_weights, grad_biases = self.get_grads(f_t_train, f_v_train, u_i_train, u_t_train)
            grads = grad_weights + grad_biases
            vars = self.weights + self.biases
            self.optimizer.apply_gradients(zip(grads, vars))

    def get_grads(self, f_t_train, f_v_train, u_i_train, u_t_train):
        with tf.GradientTape(persistent=True) as gtu:
            u_i_predict = self.i(u_t_train)
            u_loss = tf.reduce_mean(tf.square(u_i_predict - u_i_train))

            f_predict = self.f(f_t_train, f_v_train)
            f_loss = tf.reduce_mean(tf.square(f_predict))

            total_loss = u_loss + f_loss
        grad_weights = gtu.gradient(total_loss, self.weights)
        grad_biases = gtu.gradient(total_loss, self.biases)

        return grad_weights, grad_biases
