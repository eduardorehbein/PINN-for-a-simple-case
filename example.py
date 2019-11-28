from pinn import CircuitPINN
from validator import CircuitCrossValidator

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Data reading
noisy_df = pd.read_csv('./matlab/noisy_t_i_v_v3.csv')
noisy_df.drop(columns='v')
df = pd.read_csv('./matlab/t_i_v_v3.csv')
df = df.join(noisy_df.set_index('t'), on='t')
shuffled_df = df.sample(frac=1)

# Setting u data (real) and f data (simulated)
train_u_percent = 0.01
u_data_len = int(train_u_percent * len(shuffled_df))
u_df = shuffled_df.sample(n=u_data_len)
f_df = shuffled_df - u_df

np_u_t = np.array([u_df['t'].values])
np_u_i = np.array([u_df['noisy_i'].values])
np_f_t = np.array([f_df['t'].values])
np_f_v = np.array([f_df['v'].values])

# PINN instancing
R = 3
L = 3
hidden_layers = [9]
learning_rate = 0.001
model = CircuitPINN(R, L, hidden_layers, learning_rate)

# PINN validation
validator = CircuitCrossValidator()
epochs = 2000
validator.validate(model, epochs, np_u_t, np_u_i, np_f_t, np_f_v)

# PINN final training  TODO: Update train usage here
# model.train(np_t, np_i, np_v, epochs=epochs)

# PINN: i response in time
ordered_t = np.array([df['t'].values])
ordered_i = np.array([df['i'].values])

prediction = model.predict(ordered_t)

# Plot the data
plt.plot(ordered_t[0], prediction.numpy()[0], label='Predicted')
plt.plot(ordered_t[0], ordered_i[0], label='Sampled')

# Add a legend
plt.legend()

# Show the plot
plt.show()
