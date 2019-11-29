import numpy as np
import copy


class CircuitCrossValidator:
    def __init__(self, n_sections=5):
        self.t_student_dict = {3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
        self.n_sections = n_sections

    def validate(self, model, epochs, np_u_t, np_u_i, np_f_t, np_f_v, np_noiseless_u_i, np_f_i):
        data_sets = self.split(np_u_t, np_u_i, np_f_t, np_f_v, np_noiseless_u_i, np_f_i)

        initial_weights = copy.deepcopy(model.weights)
        initial_biases = copy.deepcopy(model.biases)

        accuracies = list()
        for index in range(len(data_sets)):
            test_data = data_sets[index]
            train_data = [data_set for data_set in data_sets if data_set != test_data]

            model.train(*train_data[:-2], epochs)
            np_test_t = np.array(test_data[0] + test_data[2])
            np_test_i = np.array(test_data[-1] + test_data[-2])
            np_prediction = model.predict(np_test_t)

            accuracy = np.mean(np.abs((np_test_i - np_prediction)/np_test_i))
            accuracies.append(accuracy)

            model.weights = initial_weights
            model.biases = initial_biases

        np_accuracies = np.array(accuracies)
        mean = np_accuracies.mean(0)
        repeatability = self.t_student()*np_accuracies.std(0)

        print('Mean error: [', 100*(mean - repeatability), '% -', 100*(mean + repeatability), '%]')

    def split(self, np_u_t, np_u_i, np_f_t, np_f_v, np_noiseless_u_i, np_f_i):
        return list(zip(np.array_split(np_u_t, self.n_sections),
                        np.array_split(np_u_i, self.n_sections),
                        np.array_split(np_f_t, self.n_sections),
                        np.array_split(np_f_v, self.n_sections),
                        np.array_split(np_noiseless_u_i, self.n_sections),
                        np.array_split(np_f_i, self.n_sections)))

    def t_student(self):
        return self.t_student_dict[self.n_sections]