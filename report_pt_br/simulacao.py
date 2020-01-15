from pinn_v1 import CircuitPINN

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Leitura de dados
df1 = pd.read_csv('../matlab/t_i_v_v3.csv')
df2 = pd.read_csv('../matlab/noisy_t_i_v_v3.csv').drop(columns=['t', 'v'])
df = pd.concat([df1, df2], axis=1, sort=False)
print(df)
shuffled_df = df.sample(frac=1)

# Separação dos dados de treino e teste
test_percent = 0.2
test_data_len = int(test_percent * len(shuffled_df))
test_shuffled_df = shuffled_df.sample(n=test_data_len)

train_shuffled_df = shuffled_df[~shuffled_df.isin(test_shuffled_df)].dropna()
train_t = np.array([train_shuffled_df['t'].values])
train_i = np.array([train_shuffled_df['noisy_i'].values])
train_v = np.array([train_shuffled_df['v'].values])

# Instanciando PINN
R = 3
L = 3
hidden_layers = [9]
learning_rate = 0.001
model = CircuitPINN(R, L, hidden_layers, learning_rate)

# Treino e teste
epochs = 15000
model.train(train_t, train_i, train_v, epochs)

sorted_test_df = test_shuffled_df.sort_values(['t'])
np_test_t = np.array([sorted_test_df['t'].values])
np_test_i = sorted_test_df['i'].values

np_prediction = model.predict(np_test_t)

# Plotando resultados
plt.title('Comparação nn(t) e i(t)')
plt.plot(np_test_t[0], np_test_i, label='i(t)')
plt.plot(np_test_t[0], np_prediction[0], label='nn(t)')
plt.legend()
plt.show()
