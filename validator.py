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

        r2s = list()
        predictions = list()
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
            predictions.append((np_test_t, np_prediction))

            np_sum_1 = np.sum(np.square(np_test_i - np_prediction))
            np_sum_2 = np.sum(np.square(np_test_i - np.mean(np_test_i)))
            r2 = 1 - np_sum_1 / np_sum_2
            print('R2 for test index =', index, '->', r2)
            r2s.append(r2)

            model.weights = copy.deepcopy(initial_weights)
            model.biases = copy.deepcopy(initial_biases)

        np_r2s = np.array(r2s)
        r2s_mean = np_r2s.mean(0)
        r2s_repeatability = self.t_student()*np_r2s.std(0)

        print('R2 correlation factor: [', r2s_mean - r2s_repeatability, '-', r2s_mean + r2s_repeatability, ']')

        return_t_pred = [np.array([]), np.array([])]
        for np_t, np_pred in predictions:
            return_t_pred[0] = np.append(return_t_pred[0], np_t)
            return_t_pred[1] = np.append(return_t_pred[1], np_pred)
        return return_t_pred

    def split(self, np_u_t, np_u_i, np_f_t, np_f_v, np_noiseless_u_i, np_f_i):
        return list(zip(np.array_split(np_u_t, self.n_sections),
                        np.array_split(np_u_i, self.n_sections),
                        np.array_split(np_f_t, self.n_sections),
                        np.array_split(np_f_v, self.n_sections),
                        np.array_split(np_noiseless_u_i, self.n_sections),
                        np.array_split(np_f_i, self.n_sections)))

    def t_student(self):
        return self.t_student_dict[self.n_sections]