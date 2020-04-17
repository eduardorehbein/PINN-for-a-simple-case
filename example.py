from pinn import CircuitPINN
from validator import PlotValidator
import pandas as pd
import numpy as np

# Data reading
train_df = pd.read_csv('./matlab/noisy_t_i_v_v5.csv')
test_df = pd.read_csv('./matlab/noisy_t_i_v_v6.csv')

# Train-test split
np_train_t = train_df['t'].values
np_train_v = train_df['v'].values
np_train_ic = train_df[train_df['t'] == 0]['i'].values

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
hidden_layers = [9]
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
model.train(np_train_t, np_train_v, np_train_ic, epochs)

# PINN testing
np_prediction = model.predict(np_test_t, np_test_v, np_test_ic)
PlotValidator.compare(np_test_t, np_test_i, np_prediction)
