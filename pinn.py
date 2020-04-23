import copy
import tensorflow as tf
import numpy as np

# TODO: Improve it basing on https://github.com/pierremtb/PINNs-TF2.0/blob/master/utils/neuralnetwork.py


class CircuitPINN:
    def __init__(self, R, L, hidden_layers, learning_rate, prediction_period, np_t_resolution, np_v_resolution):
        # Circuit parameters
        self.R = R  # Resistance
        self.L = L  # Inductance

        # Network's prediction period, nn(t, v, ic) works for ts in [0, prediction period]
        self.prediction_period = prediction_period

        # Time and voltage signal resolution
        self.np_t_resolution = np_t_resolution
        self.np_v_resolution = np_v_resolution

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
        subsamples = self.slash_sample(np_prediction_t, np_prediction_v, append_ic=False, slash_on_v_change=False)

        predictions = []
        np_ic_value = np_prediction_ic
        for subsample in subsamples:
            t_and_v_tuple = [*zip(*subsample)]
            np_ic = np.repeat(np_ic_value, len(t_and_v_tuple[0]))

            tf_x = tf.constant(np.array([t_and_v_tuple[0], t_and_v_tuple[1], np_ic]), dtype=tf.float32)
            np_nn = self.nn(tf_x).numpy()[0]
            np_ic_value = np_nn[-1]
            predictions.append(np_nn)

        return np.concatenate(predictions)

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

    def slash_sample(self, np_sample_t, np_sample_v, append_ic, slash_on_v_change):
        subsamples = [[]]
        np_last_transition_t = np_sample_t[0]
        np_last_v = np_sample_v[0]
        subsamples[-1].append((np.array(0), np.array(np_last_v)))
        for np_t, np_v in np.nditer([np_sample_t[1:], np_sample_v[1:]]):
            np_t_mark = np.array(np_t - np_last_transition_t)
            last_sample = subsamples[-1]
            last_sample.append((np_t_mark, np_v))
            if slash_on_v_change:
                v_condition = np.abs(np_v - np_last_v) >= self.np_v_resolution / 2
            else:
                v_condition = False
            if (v_condition or np.abs(self.prediction_period - np_t_mark) <= self.np_t_resolution / 2) \
                    and np_t != np_sample_t[-1]:
                np_last_v = np_v
                np_last_transition_t = np_t
                subsamples.append([])
                if append_ic:
                    last_sample = subsamples[-1]
                    last_sample.append((np.array(0), np_v))

        return subsamples

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
