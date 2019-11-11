from pinn import CircuitPINN

import pandas as pd
import numpy as np
import tensorflow as tf

# Data reading
df = pd.read_csv('./cleared_t_i_v.csv')
# df = pd.read_csv('./matlab/t_i_v_v2.csv')

# Train and test data splitting
# Obs: data is already shuffled
train_index = int(0.8*len(df))
train_df = df[:train_index]
test_df = df[train_index:]

train_t = np.array([train_df['t'].values])
train_i = np.array([train_df['i'].values])
train_v = np.array([train_df['v'].values])

test_t = np.array([test_df['t'].values])
test_i = np.array([test_df['i'].values])

# PINN instancing
R = 1000
L = 0.001
hidden_layers = [5, 5, 5]
learning_rate = 0.0005

circuit = CircuitPINN(R, L, hidden_layers, learning_rate)

# PINN training
circuit.train(train_t, train_i, train_v, epochs=1500)

# PINN testing
prediction = circuit.predict(test_t)
print(tf.reduce_mean(tf.abs((test_i - prediction)/test_i)))
