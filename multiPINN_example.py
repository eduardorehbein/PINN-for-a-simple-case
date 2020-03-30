from pinn import HighLevelRLCircuitPINN
import numpy as np

# Data generating
t = [0.01*j for j in range(701)]

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


initial_condition_t = 0
initial_condition_i = 0

np_subpinns_train_u_t = np.array([initial_condition_t for j in range(len(subpinns_train_vs))])
np_subpinns_train_u_v = np.array(subpinns_train_vs)
np_subpinns_train_u_i = np.array([initial_condition_i for j in range(len(subpinns_train_vs))])
np_subpinns_train_f_t = get_np_t_from(subpinns_train_vs)
np_subpinns_train_f_v = get_np_v_from(subpinns_train_vs)

np_hlpinn_train_u_t = np.array([initial_condition_t for j in range(len(hlpinn_train_vs))])
np_hlpinn_train_u_v = np.array(hlpinn_train_vs)
np_hlpinn_train_u_i = np.array([initial_condition_i for j in range(len(hlpinn_train_vs))])
np_hlpinn_train_f_t = get_np_t_from(hlpinn_train_vs)
np_hlpinn_train_f_v = get_np_v_from(hlpinn_train_vs)

np_hlpinn_test_f_t = get_np_t_from(hlpinn_test_vs)
np_hlpinn_test_f_v = get_np_v_from(hlpinn_test_vs)

# PINN instancing
R = 3
L = 3
hidden_layers = [9, 9]
learning_rate = 0.001

subpinns = list()

high_level_model = HighLevelRLCircuitPINN(R, L, subpinns, hidden_layers, learning_rate)

# TODO: Put initial condition for u training
