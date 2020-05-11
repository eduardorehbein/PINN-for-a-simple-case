from pinn import CircuitPINN
from normalizer import Normalizer
import numpy as np
from scipy.integrate import odeint
import random

# Circuit parameters
R = 3
L = 3

# Setting train data
random.seed(30)
t = [0.01*j for j in range(701)]

train_ics = [((-1) ** j) * 4 * random.random() for j in range(100)]
train_vs = [((-1) ** j) * 20 * random.random() for j in range(len(train_ics))]

random.shuffle(train_ics)
random.shuffle(train_vs)

np_train_u_t = np.zeros(len(train_ics))
np_train_u_v = np.array(train_vs)
np_train_u_ic = np.array(train_ics)

np_train_f_t = None
np_train_f_v = None
np_train_f_ic = None

np_t = np.array(t)
for j in range(len(train_vs)):
    np_ic = np.full((len(t),), train_ics[j])
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

# Normalizers
t_normalizer = Normalizer()
v_normalizer = Normalizer()
i_normalizer = Normalizer()

t_normalizer.parametrize(np_t)
v_normalizer.parametrize(np.array(train_vs))
i_normalizer.parametrize(np.array(train_ics))

# Train data normalization
np_norm_train_u_t = t_normalizer.normalize(np_train_u_t)
np_norm_train_u_v = v_normalizer.normalize(np_train_u_v)
np_norm_train_u_ic = i_normalizer.normalize(np_train_u_ic)

np_norm_train_f_t = t_normalizer.normalize(np_train_f_t)
np_norm_train_f_v = v_normalizer.normalize(np_train_f_v)
np_norm_train_f_ic = i_normalizer.normalize(np_train_f_ic)

# PINN instancing
hidden_layers = [9, 9]
learning_rate = 0.001

# Model for normalized data
model = CircuitPINN(R=R,
                    L=L,
                    hidden_layers=hidden_layers,
                    learning_rate=learning_rate,
                    t_normalizer=t_normalizer,
                    v_normalizer=v_normalizer,
                    i_normalizer=i_normalizer)

# Model for denormalized data
# model = CircuitPINN(R=R, L=L, hidden_layers=hidden_layers, learning_rate=learning_rate)

# PINN training
max_epochs = 15000
stop_loss = 0.0002

# Train with normalized data
# model.train(np_norm_train_u_t, np_norm_train_u_v, np_norm_train_u_ic, np_norm_train_f_t, np_norm_train_f_v,
#             np_norm_train_f_ic, max_epochs=max_epochs, stop_loss=stop_loss)

# Train with denormalized data
# model.train(np_train_u_t, np_train_u_v, np_train_u_ic, np_train_f_t, np_train_f_v, np_train_f_ic, epochs)

# Setting test data
test_vs = [10, 12, 7, 4, 8, 11, 13, -1, -6]
test_ics = [((-1) ** j) * 4 * random.random() for j in range(len(test_vs))]

sampled_outputs = []
predictions = []

np_norm_t = t_normalizer.normalize(np_t)
for j in range(len(test_vs)):
    test_v = test_vs[j]
    test_ic = test_ics[j]

    np_i = np.transpose(odeint(lambda i_t, time_t: (1/L) * test_v - (R / L) * i_t, test_ic, np_t))[0]
    sampled_outputs.append(np_i)

    # PINN testing
    np_ic = np.full((len(t),), test_ic)
    np_v = np.full((len(t),), test_v)

    np_norm_ic = i_normalizer.normalize(np_ic)
    np_norm_v = v_normalizer.normalize(np_v)

    # Test with normalized data
    np_norm_prediction = model.predict(np_norm_t, np_norm_v, np_norm_ic)
    np_prediction = i_normalizer.denormalize(np_norm_prediction)

    # Test with denormalized data
    # np_prediction = model.predict(np_t, np_v, np_ic)

    predictions.append(np_prediction)

# Print Results
test_mse = (np.square(np.array(sampled_outputs) - np.array(predictions))).mean()
print('Test MSE: ' + str(test_mse))
