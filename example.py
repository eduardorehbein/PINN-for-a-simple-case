from pinn import CircuitPINN
from validator import PlotValidator
import numpy as np
from scipy.integrate import odeint
import random

# Train data
random.seed(30)

t = [0.01*j for j in range(701)]
initial_conditions = [(-1**j)*4*random.random() for j in range(100)]
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
R = 3
L = 3

test_vs = [10, 12, 7, 8, 4, 8, 11, 13, 11, -1, -6, -3, 2]


def v_t(t1):
    if t1 <= 7:
        return test_vs[0]
    elif t1 <= 14:
        return test_vs[1]
    elif t1 <= 21:
        return test_vs[2]
    elif t1 <= 28:
        return test_vs[3]
    elif t1 <= 35:
        return test_vs[4]
    elif t1 <= 42:
        return test_vs[5]
    elif t1 <= 49:
        return test_vs[6]
    elif t1 <= 56:
        return test_vs[7]
    elif t1 <= 63:
        return test_vs[8]
    elif t1 <= 70:
        return test_vs[9]
    elif t1 <= 77:
        return test_vs[10]
    elif t1 <= 84:
        return test_vs[11]
    else:
        return test_vs[12]


def di_dt(i_t, t1):
    return (1/L)*v_t(t1) - (R/L)*i_t


np_test_t = np.array([0.01*j for j in range(9101)])
np_test_v = np.array([v_t(t1) for t1 in np_test_t])
np_test_ic = np.array([0])
np_test_i = odeint(di_dt, np_test_ic, np_test_t)

np_t_resolution = np.array(0.01)
np_v_resolution = np.array(1)

# PINN instancing
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
