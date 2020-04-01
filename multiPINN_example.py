from pinn import CircuitPINN, HighLevelRLCircuitPINN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data generating
t = [0.01*j for j in range(701)]
initial_condition_t = 0
initial_condition_i = 0

subpinns_train_vs = [-10, -6, -2, 2, 6, 10]
hlpinn_train_vs = [-8, -4, 0, 4, 8]
hlpinn_test_vs = [-9, -7, -5, -3, -1, 1, 3, 5, 7, 9]


def get_np_t_from(vs_list):
    t_list = list()
    for j in range(len(vs_list)):
        t_list = t_list + t

    return np.array(t_list)


def get_np_v_from(vs_list):
    v_list = list()
    for v in vs_list:
        v_list = v_list + [v for j in range(len(t))]

    return np.array(v_list)


np_subpinn_train_u_t = np.array([initial_condition_t])
np_subpinn_train_u_i = np.array([initial_condition_i])
np_subpinn_train_f_t = np.array(t)
np.random.shuffle(np_subpinn_train_f_t)

np_hlpinn_train_u_t = np.array([initial_condition_t for j in range(len(hlpinn_train_vs))])
np_hlpinn_train_u_v = np.array(hlpinn_train_vs)
np_hlpinn_train_u_i = np.array([initial_condition_i for j in range(len(hlpinn_train_vs))])
np_hlpinn_train_f_t = get_np_t_from(hlpinn_train_vs)
np_hlpinn_train_f_v = get_np_v_from(hlpinn_train_vs)
df_hlpinn_train = pd.DataFrame(data={'f_t': np_hlpinn_train_f_t, 'f_v': np_hlpinn_train_f_v})
shuffled_df_hlpinn_train = df_hlpinn_train.sample(frac=1)
np_hlpinn_train_f_t = shuffled_df_hlpinn_train['f_t'].values
np_hlpinn_train_f_v = shuffled_df_hlpinn_train['f_v'].values

df_hlpinn_test = pd.read_csv('./matlab/multi_pinn.csv')
np_hlpinn_test_t = df_hlpinn_test['t'].values
np_hlpinn_test_v = df_hlpinn_test['v'].values
np_hlpinn_test_i = df_hlpinn_test['i'].values

# PINNs' params
R = 3
L = 3
hidden_layers = [9]
learning_rate = 0.001
epochs = 1000

subpinns = list()
for subpinn_train_v in subpinns_train_vs:
    subpinn = CircuitPINN(R, L, hidden_layers, learning_rate)
    np_subpinn_train_f_v = np.array([subpinn_train_v for j in range(len(t))])
    subpinn.train(np_subpinn_train_u_t, np_subpinn_train_u_i, np_subpinn_train_f_t, np_subpinn_train_f_v, epochs)
    subpinns.append(subpinn)

hidden_layers = [9, 9]
high_level_model = HighLevelRLCircuitPINN(R, L, subpinns, hidden_layers, learning_rate)
high_level_model.train(np_hlpinn_train_u_t, np_hlpinn_train_u_v, np_hlpinn_train_u_i, np_hlpinn_train_f_t,
                       np_hlpinn_train_f_v, epochs)

np_test_NN = high_level_model.predict(np_hlpinn_test_t, np_hlpinn_test_v)

# Plot the data
plt.plot(np_hlpinn_test_t, np.transpose(np_test_NN), label='Predicted')
plt.plot(np_hlpinn_test_t, np_hlpinn_test_i, label='Sampled')

# Add a legend
plt.legend()

# Show the plot
plt.show()
