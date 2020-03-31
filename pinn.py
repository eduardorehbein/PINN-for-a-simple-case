import tensorflow as tf
import numpy as np
import copy

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

    def predict(self, np_t):
        tf_t = tf.constant(np.array([np_t]), dtype=tf.float32)
        tf_i = self.i(tf_t)

        return tf_i.numpy()[0]

    def i(self, tf_t):
        num_layers = len(self.weights) + 1
        tf_U = tf_t
        for l in range(0, num_layers - 2):
            tf_W = self.weights[l]
            tf_b = self.biases[l]
            tf_U = tf.tanh(tf.add(tf.matmul(tf_W, tf_U), tf_b))
        tf_W = self.weights[-1]
        tf_b = self.biases[-1]
        tf_Y = tf.add(tf.matmul(tf_W, tf_U), tf_b)

        return tf_Y

    def f(self, tf_t, tf_v):
        # ODE: Ldi_dt + Ri = v
        with tf.GradientTape() as gtf:
            gtf.watch(tf_t)
            tf_i = self.i(tf_t)
        tf_di_dt = gtf.gradient(tf_i, tf_t)

        return tf_di_dt + (self.R / self.L) * tf_i - (1 / self.L) * tf_v

    def train(self, np_u_t, np_u_i, np_f_t, np_f_v, epochs=1):
        # Data for u loss
        tf_u_t = tf.constant(np.array([np_u_t]), dtype=tf.float32)
        tf_u_i = tf.constant(np.array([np_u_i]), dtype=tf.float32)

        # Data for f loss
        tf_f_t = tf.constant(np.array([np_f_t]), dtype=tf.float32)
        tf_f_v = tf.constant(np.array([np_f_v]), dtype=tf.float32)

        for j in range(epochs):
            # Gradients
            grad_weights, grad_biases = self.get_grads(tf_u_t, tf_u_i, tf_f_t, tf_f_v)

            # Updating weights and biases
            grads = grad_weights + grad_biases
            vars_to_update = self.weights + self.biases
            self.optimizer.apply_gradients(zip(grads, vars_to_update))

    def get_grads(self, tf_u_t, tf_u_i, tf_f_t, tf_f_v):
        with tf.GradientTape(persistent=True) as gtu:
            tf_u_i_predict = self.i(tf_u_t)
            tf_u_loss = tf.reduce_mean(tf.square(tf_u_i_predict - tf_u_i))

            tf_f_predict = self.f(tf_f_t, tf_f_v)
            tf_f_loss = tf.reduce_mean(tf.square(tf_f_predict))

            tf_total_loss = tf_u_loss + tf_f_loss
        grad_weights = gtu.gradient(tf_total_loss, self.weights)
        grad_biases = gtu.gradient(tf_total_loss, self.biases)

        return grad_weights, grad_biases


class HighLevelRLCircuitPINN:
    def __init__(self, R, L, subpinns, hidden_layers, learning_rate):
        # Circuit parameters
        self.R = R  # Resistance
        self.L = L  # Inductance

        # Initialize NN
        self.layers = [len(subpinns) + 2] + hidden_layers + [1]
        self.weights, self.biases = self.initialize_NN(self.layers)
        self.initial_weights, self.initial_biases = copy.deepcopy(self.weights), copy.deepcopy(self.biases)

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

    def predict(self, np_t, np_v):
        tf_x = tf.constant(np.array([np_t, np_v]), dtype=tf.float32)
        tf_NN = self.NN(tf_x)

        return tf_NN.numpy()

    def NN(self, tf_x):
        """
        Network feeding.

        :param tf_x: (t, v)
        :return: i
        """

        # TODO: Get t from x
        # TODO: Feed each subpinn with t
        # TODO: Concat x with subpinns' output (tf_U)

        num_layers = len(self.weights) + 1
        tf_U = tf_x
        for l in range(0, num_layers - 2):
            tf_W = self.weights[l]
            tf_b = self.biases[l]
            tf_U = tf.tanh(tf.add(tf.matmul(tf_W, tf_U), tf_b))
        tf_W = self.weights[-1]
        tf_b = self.biases[-1]
        tf_Y = tf.add(tf.matmul(tf_W, tf_U), tf_b)

        return tf_Y

    def f(self, tf_x, tf_v):
        # ODE: Ldi_dt + Ri = v
        with tf.GradientTape() as gtf:
            gtf.watch(tf_x)
            tf_NN = self.NN(tf_x)
        tf_dNN_dx = gtf.gradient(tf_NN, tf_x)  # TODO: Look at newer versions to see how to separate t and v

        return tf_dNN_dt + (self.R / self.L) * tf_NN - (1 / self.L) * tf_v

    def train(self, np_u_t, np_u_v, np_u_i, np_f_t, np_f_v, epochs=1):
        """
        PINN training
        :param int epochs: number of training epochs
        :return:
        """
        # Data for u loss
        tf_u_x = tf.constant(np.array([np_u_t, np_u_v]), dtype=tf.float32)
        tf_u_i = tf.constant(np.array([np_u_i]), dtype=tf.float32)

        # Data for f loss
        tf_f_x = tf.constant(np.array([np_f_t, np_f_v]), dtype=tf.float32)
        tf_f_v = tf.constant(np.array([np_f_v]), dtype=tf.float32)

        for j in range(epochs):
            # Gradients
            grad_weights, grad_biases = self.get_grads(tf_u_x, tf_u_i, tf_f_x, tf_f_v)

            # Updating weights and biases
            grads = grad_weights + grad_biases
            vars_to_update = self.weights + self.biases
            self.optimizer.apply_gradients(zip(grads, vars_to_update))

    def get_grads(self, tf_u_x, tf_u_i, tf_f_x, tf_f_v):
        with tf.GradientTape(persistent=True) as gt:
            tf_u_i_predict = self.NN(tf_u_x)
            tf_u_loss = tf.reduce_mean(tf.square(tf_u_i_predict - tf_u_i))

            tf_f_predict = tf.square(self.f(tf_f_x, tf_f_v))
            tf_f_loss = tf.reduce_mean(tf_f_predict)

            tf_total_loss = tf_u_loss + tf_f_loss
        grad_weights = gt.gradient(tf_total_loss, self.weights)
        grad_biases = gt.gradient(tf_total_loss, self.biases)

        return grad_weights, grad_biases

    def reset_NN(self):
        self.weights = copy.deepcopy(self.initial_weights)
        self.biases = copy.deepcopy(self.initial_biases)
