from pinn_v1 import CircuitPINN

import matplotlib.pyplot as plt
import pandas as pd

# Leitura de dados
df = pd.read_csv('../matlab/noisy_t_i_v_v3.csv')
shuffled_df = df.sample(frac=1)

# Separação dos dados de treino e teste
test_percent = 0.2
test_data_len = int(test_percent * len(shuffled_df))
test_shuffled_df = shuffled_df.sample(n=test_data_len)
train_shuffled_df = shuffled_df[~shuffled_df.isin(test_shuffled_df)].dropna()

# Preparação dos dados reais e simulados
train_u_percent = 0.01
u_data_len = int(train_u_percent * len(train_shuffled_df))
u_df = train_shuffled_df.sample(n=u_data_len)
f_df = train_shuffled_df[~train_shuffled_df.isin(u_df)].dropna()

# Conversão para dados numpy
np_u_t = u_df['t'].values
np_u_i = u_df['noisy_i'].values
np_f_t = f_df['t'].values
np_f_v = f_df['v'].values

# Instanciando PINN
R = 3
L = 3
hidden_layers = [9]
learning_rate = 0.001
model = CircuitPINN(R, L, hidden_layers, learning_rate)

# Treino e teste
epochs = 15000
model.train(np_u_t, np_u_i, np_f_t, np_f_v, epochs)

sorted_test_df = test_shuffled_df.sort(['t'])
np_test_t = sorted_test_df['t'].values
np_test_i = sorted_test_df['i'].values

np_prediction = model.predict(np_test_t)

# Plotando resultados
plt.title('Comparação nn(t) e i(t)')
plt.plot(np_test_t, np_test_i, label='i(t)')
plt.plot(np_test_t, np_prediction, label='nn(t)')
plt.legend()
plt.show()
