from pinn import CircuitPINN

import matplotlib.pyplot as plt
import pandas as pd

# Data reading
df = pd.read_csv('./matlab/noisy_t_i_v_v7.csv')
shuffled_df = df.sample(frac=1)

# Train, test, validation split
validation_v_steps = [3, 5, 12, 20]
validation_df = df[df['v'].isin(validation_v_steps)]
train_test_shuffled_df = shuffled_df[~shuffled_df.isin(validation_df)].dropna()

# Setting u data (real) and f data (simulated)
train_u_percent = 0.01
u_data_len = int(train_u_percent * len(train_test_shuffled_df))
u_df = train_test_shuffled_df.sample(n=u_data_len)
f_df = train_test_shuffled_df[~train_test_shuffled_df.isin(u_df)].dropna()

# Converting to numpy data
np_u_t = u_df['t'].values
np_u_v = u_df['v'].values
np_u_i = u_df['noisy_i'].values
np_noiseless_u_i = u_df['i'].values
np_f_t = f_df['t'].values
np_f_v = f_df['v'].values
np_f_i = f_df['i'].values

# PINN instancing
R = 3
L = 3
hidden_layers = [9]
learning_rate = 0.001
model = CircuitPINN(R, L, hidden_layers, learning_rate)

# PINN validation
epochs = 15000
model.train(np_u_t, np_u_v, np_u_i, np_f_t, np_f_v, epochs)

for index, v_step in enumerate(validation_v_steps):
    single_step_validation_df = validation_df[validation_df['v'] == v_step]
    np_validation_t = single_step_validation_df['t'].values
    np_validation_v = single_step_validation_df['v'].values
    np_validation_i = single_step_validation_df['i'].values
    np_prediction = model.predict(np_validation_t, np_validation_v)

    plt.subplot(2, 2, index + 1)
    plt.title('Sampled vs predicted i(t) for v(t) = ' + str(v_step) + 'D(t)')
    plt.plot(np_validation_t, np_validation_i, label='Sampled i(t)')
    plt.plot(np_validation_t, np_prediction, label='Predicted i(t)')
    plt.legend()

plt.show()
