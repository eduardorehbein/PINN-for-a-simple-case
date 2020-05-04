import copy
import tensorflow as tf
import numpy as np

# TODO: Improve it basing on https://github.com/pierremtb/PINNs-TF2.0/blob/master/utils/neuralnetwork.py


class CircuitPINN:
    def __init__(self, R, L, hidden_layers, learning_rate, t_normalizer=None, v_normalizer=None, i_normalizer=None):
        # Circuit parameters
        self.R = R  # Resistance
        self.L = L  # Inductance

        # Data normalizers
        self.t_normalizer = t_normalizer
        self.v_normalizer = v_normalizer
        self.i_normalizer = i_normalizer

        if self.t_normalizer is None or self.v_normalizer is None or self.i_normalizer is None:
            self.data_is_normalized = False
        else:
            self.data_is_normalized = True

        # Initialize NN
        self.layers = [3] + hidden_layers + [1]
        self.weights, self.biases = self.initialize_nn(self.layers)
        self.initial_weights, self.initial_biases = copy.deepcopy(self.weights), copy.deepcopy(self.biases)

        # Optimizer
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    def initialize_nn(self, layers):
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

    def predict(self, np_prediction_t, np_prediction_v, np_prediction_ic):
        tf_x = tf.constant(np.array([np_prediction_t, np_prediction_v, np_prediction_ic]), dtype=tf.float32)

        return self.nn(tf_x).numpy()[0]

    def nn(self, tf_x):
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
            tf_nn = self.nn(tf_x)
        tf_dnn_dx = gtf.gradient(tf_nn, tf_x)
        tf_dnn_dt = tf.slice(tf_dnn_dx, [0, 0], [1, tf_dnn_dx.shape[1]])
        if self.data_is_normalized:
            return (self.i_normalizer.std / self.t_normalizer.std) * tf_dnn_dt + \
                   (self.R / self.L) * self.i_normalizer.denormalize(tf_nn) - \
                   (1 / self.L) * self.v_normalizer.denormalize(tf_v)
        else:
            return tf_dnn_dt + (self.R / self.L) * tf_nn - (1 / self.L) * tf_v

    def train(self, np_u_t, np_u_v, np_u_ic, np_f_t, np_f_v, np_f_ic, epochs=1):
        tf_u_x = tf.constant(np.array([np_u_t, np_u_v, np_u_ic]), dtype=tf.float32)
        tf_u_ic = tf.constant(np.array([np_u_ic]), dtype=tf.float32)

        np_f_x = np.array([np_f_t, np_f_v, np_f_ic])
        np.random.shuffle(np.transpose(np_f_x))

        tf_f_x = tf.constant(np.array(np_f_x), dtype=tf.float32)
        tf_f_v = tf.constant(np.array([np_f_x[1]]), dtype=tf.float32)

        for j in range(epochs):
            # Gradients
            grad_weights, grad_biases = self.get_grads(tf_u_x, tf_u_ic, tf_f_x, tf_f_v)

            # Updating weights and biases
            grads = grad_weights + grad_biases
            vars_to_update = self.weights + self.biases
            self.optimizer.apply_gradients(zip(grads, vars_to_update))

    def get_grads(self, tf_u_x, tf_u_ic, tf_f_x, tf_f_v):
        with tf.GradientTape(persistent=True) as gtu:
            tf_u_predict = self.nn(tf_u_x)
            tf_u_loss = tf.reduce_mean(tf.square(tf_u_predict - tf_u_ic))

            tf_f_predict = self.f(tf_f_x, tf_f_v)
            tf_f_loss = tf.reduce_mean(tf.square(tf_f_predict))

            tf_total_loss = tf_u_loss + tf_f_loss
        grad_weights = gtu.gradient(tf_total_loss, self.weights)
        grad_biases = gtu.gradient(tf_total_loss, self.biases)

        return grad_weights, grad_biases

    def reset_NN(self):
        self.weights = copy.deepcopy(self.initial_weights)
        self.biases = copy.deepcopy(self.initial_biases)
