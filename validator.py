import numpy as np
import copy
import matplotlib.pyplot as plt


class CircuitCrossValidator:
    def __init__(self, n_sections=5):
        self.t_student_dict = {3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
        self.n_sections = n_sections

    def validate(self, model, epochs, np_u_t, np_u_i, np_f_t, np_f_v, np_noiseless_u_i, np_f_i):
        data_sets = self.split(np_u_t, np_u_i, np_f_t, np_f_v, np_noiseless_u_i, np_f_i)

        initial_weights = copy.deepcopy(model.weights)
        initial_biases = copy.deepcopy(model.biases)

        mses = list()
        for index in range(len(data_sets)):
            test_data = data_sets[index]

            train_data_sets = data_sets[:index] + data_sets[index+1:]
            train_data = [np.array([]), np.array([]), np.array([]), np.array([])]
            for train_data_set in train_data_sets:
                for j, data in enumerate(train_data_set):
                    if j < 4:
                        train_data[j] = np.append(train_data[j], data)
            model.train(*train_data, epochs)
            np_test_t = np.append(test_data[0], test_data[2])
            np_test_i = np.append(test_data[-1], test_data[-2])
            np_prediction = model.predict(np_test_t)

            mse = np.mean(np.square(np_prediction - np_test_i))
            mses.append(mse)

            model.weights = copy.deepcopy(initial_weights)
            model.biases = copy.deepcopy(initial_biases)

        np_mses = np.array(mses)
        mean = np_mses.mean(0)
        repeatability = self.t_student()*np_mses.std(0)

        print('MSE error: [', mean - repeatability, '-', mean + repeatability, ']')

    def split(self, np_u_t, np_u_i, np_f_t, np_f_v, np_noiseless_u_i, np_f_i):
        return list(zip(np.array_split(np_u_t, self.n_sections),
                        np.array_split(np_u_i, self.n_sections),
                        np.array_split(np_f_t, self.n_sections),
                        np.array_split(np_f_v, self.n_sections),
                        np.array_split(np_noiseless_u_i, self.n_sections),
                        np.array_split(np_f_i, self.n_sections)))

    def t_student(self):
        return self.t_student_dict[self.n_sections]