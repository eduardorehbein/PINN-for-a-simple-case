from pinn import CircuitPINN
from validator import CircuitCrossValidator

import matplotlib.pyplot as plt
import pandas as pd

# Data reading
df = pd.read_csv('./matlab/noisy_t_i_v_v4.csv')
shuffled_df = df.sample(frac=1)

# Setting u data (real) and f data (simulated)
train_u_percent = 0.01
u_data_len = int(train_u_percent * len(shuffled_df))
u_df = shuffled_df.sample(n=u_data_len)
f_df = shuffled_df[~shuffled_df.isin(u_df)].dropna()

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
hidden_layers = [5]
learning_rate = 0.001
model = CircuitPINN(R, L, hidden_layers, learning_rate)

# PINN validation

losses = list()

n_sections = 4
epochs = 8000

validator1 = CircuitCrossValidator(n_sections)
validator1.validate(model, epochs, np_u_t, np_u_v, np_u_i, np_f_t, np_f_v, np_noiseless_u_i, np_f_i)
losses.append(validator1.np_mse_total_loss)

# -------------------------------------------------------------------------------------------------------
# Setting u data (real) and f data (simulated)
train_u_percent = 0.99
u_data_len = int(train_u_percent * len(shuffled_df))
u_df = shuffled_df.sample(n=u_data_len)
f_df = shuffled_df[~shuffled_df.isin(u_df)].dropna()

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
hidden_layers = [5]
learning_rate = 0.001
model = CircuitPINN(R, L, hidden_layers, learning_rate)

# PINN validation
n_sections = 4
epochs = 8000

validator2 = CircuitCrossValidator(n_sections)
validator2.validate(model, epochs, np_u_t, np_u_v, np_u_i, np_f_t, np_f_v, np_noiseless_u_i, np_f_i)
losses.append(validator2.np_mse_total_loss)

plt.title('Losses')
plt.plot(range(len(losses[0])), losses[0], label='MSE PINN')
plt.plot(range(len(losses[1])), losses[1], label='MSE MLP')
plt.legend()
plt.show()
