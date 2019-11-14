from pinn import CircuitPINN

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Data reading
df = pd.read_csv('./cleared_t_i_v.csv')
# df = (df-df.mean())/df.std() # Normalizing

t = np.array([df['t'].values])
i = np.array([df['i'].values])
v = np.array([df['v'].values])

# PINN instancing
R = 3
L = 3
hidden_layers = [9, 9]
learning_rate = 0.001

circuit = CircuitPINN(R, L, hidden_layers, learning_rate)

# PINN training
circuit.train(t, i, v, epochs=20000)

# PINN testing
ordered_df = df.sort_values(by=['t'])
ordered_t = np.array([ordered_df['t'].values])
ordered_i = np.array([ordered_df['i'].values])

prediction = circuit.predict(ordered_t)
print('MSE da predição com os dados de treino no teste',
      tf.reduce_mean(tf.square(tf.constant(ordered_i, dtype=tf.float32) - prediction)))

# Plot the data
plt.plot(ordered_t[0], prediction.numpy()[0], label='Predicted')
plt.plot(ordered_t[0], ordered_i[0], label='Sampled')

# Add a legend
plt.legend()

# Show the plot
plt.show()
