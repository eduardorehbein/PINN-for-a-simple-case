from pinn import CircuitPINN
from validator import PlotValidator
import pandas as pd
import numpy as np
import random

# Train data
random.seed(301)

t = [0.01*j for j in range(701)]
initial_conditions = [0, 0, -2, -4, 5, 3, 1.5, 1.5, 3, 2, 2.7, 1.6, 1.2, 1, -3.6, -3.6, -3.6, -2, 2, -1.1]
train_vs = [(-1**j)*20*random.random() for j in range(len(initial_conditions))]

np_train_u_t = np.zeros(len(initial_conditions))
np_train_u_v = np.array(train_vs)
np_train_u_ic = np.array(initial_conditions)

np_train_f_t = None
np_train_f_v = None
np_train_f_ic = None

np_t = np.array(t)
for j in range(len(train_vs)):
    np_ic = np.full((len(t),), initial_conditions[j])
    np_v = np.full((len(t),), train_vs[j])

    if np_train_f_t is None:
        np_train_f_t = np_t
    else:
        np_train_f_t = np.append(np_train_f_t, np_t)
    if np_train_f_v is None:
        np_train_f_v = np_v
    else:
        np_train_f_v = np.append(np_train_f_v, np_v)
    if np_train_f_ic is None:
        np_train_f_ic = np_ic
    else:
        np_train_f_ic = np.append(np_train_f_ic, np_ic)

# Test data
test_df = pd.read_csv('./matlab/noisy_t_i_v_v8.csv')

np_test_t = test_df['t'].values
np_test_v = test_df['v'].values
np_test_ic = test_df[test_df['t'] == 0]['i'].values
np_test_i = test_df['i'].values

np_t_resolution = np.array(0.01)
np_v_resolution = np.array(1)

# PINN instancing
R = 3
L = 3
prediction_period = 7
hidden_layers = [9, 9]
learning_rate = 0.001
model = CircuitPINN(R=R,
                    L=L,
                    hidden_layers=hidden_layers,
                    learning_rate=learning_rate,
                    prediction_period=prediction_period,
                    np_t_resolution=np_t_resolution,
                    np_v_resolution=np_v_resolution)

# PINN training
epochs = 10000
model.train(np_train_u_t, np_train_u_v, np_train_u_ic, np_train_f_t, np_train_f_v, np_train_f_ic, epochs)

# PINN testing
np_prediction = model.predict(np_test_t, np_test_v, np_test_ic)
PlotValidator.compare(np_test_t, np_test_i, np_prediction)
