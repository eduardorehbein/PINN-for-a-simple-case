from pinn import CircuitPINN
from validator import PlotValidator
import pandas as pd
import numpy as np

# Data reading
train_df = pd.read_csv('./matlab/noisy_t_i_v_v5.csv')
test_df = pd.read_csv('./matlab/noisy_t_i_v_v6.csv')
# shuffled_df = df.sample(frac=1)

# Train-test split
np_train_t = train_df['t'].values
np_train_v = train_df['v'].values
np_train_ic = train_df[train_df['t'] == 0]['i'].values

np_test_t = test_df['t'].values
np_test_v = test_df['v'].values
np_test_ic = test_df[test_df['t'] == 0]['i'].values

np_t_resolution = np.array(0.01)
np_v_resolution = np.array(1)

# PINN instancing
R = 3
L = 3
prediction_period = 7
hidden_layers = [9, 9]
learning_rate = 0.001
model = CircuitPINN(R, L, prediction_period, hidden_layers, learning_rate)

# PINN training
epochs = 10000
# model.train(np_train_t, np_train_v, np_train_ic, epochs)

# PINN testing
prediction = model.predict(np_test_t, np_test_v, np_test_ic, np_t_resolution, np_v_resolution)

x_axis = list()
# validation_outputs = list()
# predictions = list()
# titles = list()
# validation_labels = ['Sampled i(t)']
# prediction_labels = ['Predicted i(t)']
# for index, v_step in enumerate(validation_v_steps):
#     single_step_validation_df = validation_df[validation_df['v'] == v_step]
#     np_validation_t = single_step_validation_df['t'].values
#     np_validation_v = single_step_validation_df['v'].values
#     np_validation_i = single_step_validation_df['i'].values
#     np_prediction = model.predict(np_validation_t, np_validation_v)
#
#     x_axis.append(np_validation_t)
#     validation_outputs.append(np_validation_i)
#     predictions.append(np_prediction)
#     titles.append('Sampled vs predicted i(t) for v(t) = ' + str(v_step) + 'D(t)')
#
# PlotValidator.multicompare(x_axis, validation_outputs, predictions, titles, validation_labels, prediction_labels)
