import numpy as np
import pandas as pd
import copy


class CircuitCrossValidator:
    def __init__(self, n_sections=5):
        self.t_student_dict = {3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
        self.n_sections = n_sections

    def validate(self, model, epochs, np_u_t, np_u_v, np_u_i, np_f_t, np_f_v, np_noiseless_u_i, np_f_i):
        # TODO: Refactor for optimization and better reading
        data_sets = self.split(np_u_t, np_u_v, np_u_i, np_f_t, np_f_v, np_noiseless_u_i, np_f_i)

        r2s_and_models = list()
        samples = list()
        predictions = list()
        for index in range(len(data_sets)):
            test_data = data_sets[index]

            train_data_sets = data_sets[:index] + data_sets[index+1:]
            train_data = [np.array([]), np.array([]), np.array([]), np.array([]), np.array([])]
            for train_data_set in train_data_sets:
                for j, data in enumerate(train_data_set):
                    if j < 5:
                        train_data[j] = np.append(train_data[j], data)

            model.train(*train_data, epochs)

            np_test_t = np.append(test_data[0], test_data[3])
            np_test_v = np.append(test_data[1], test_data[4])
            np_test_i = np.append(test_data[-2], test_data[-1])

            np_prediction = model.predict(np_test_t, np_test_v)

            samples.append((np_test_t, np_test_v, np_test_i))
            predictions.append((np_test_t, np_test_v, np_prediction))

            np_sum_1 = np.sum(np.square(np_test_i - np_prediction))
            np_sum_2 = np.sum(np.square(np_test_i - np.mean(np_test_i)))
            r2 = 1 - np_sum_1 / np_sum_2
            print('R2 for test set', index + 1, '->', r2)
            r2s_and_models.append((r2, copy.deepcopy(model)))

            model.reset_NN()

        self.print_correlation_analysis(r2s_and_models)
        return self.get_best_model(r2s_and_models)

    def print_correlation_analysis(self, r2s_and_models):
        r2s = list(zip(*r2s_and_models))[0]
        np_r2s = np.array(r2s)
        r2s_mean = np_r2s.mean(0)
        r2s_repeatability = self.t_student() * np_r2s.std(0)
        print('R2 correlation factor must be in [',
              r2s_mean - r2s_repeatability, ',', r2s_mean + r2s_repeatability, ']')

    def split(self, np_u_t, np_u_v, np_u_i, np_f_t, np_f_v, np_noiseless_u_i, np_f_i):
        return list(zip(np.array_split(np_u_t, self.n_sections),
                        np.array_split(np_u_v, self.n_sections),
                        np.array_split(np_u_i, self.n_sections),
                        np.array_split(np_f_t, self.n_sections),
                        np.array_split(np_f_v, self.n_sections),
                        np.array_split(np_noiseless_u_i, self.n_sections),
                        np.array_split(np_f_i, self.n_sections)))

    def t_student(self):
        return self.t_student_dict[self.n_sections]

    def sort_by_columns(self, data, columns, columns_to_sort):
        index = ['Sample' + str(i) for i in range(1, data.shape[0] + 1)]
        df = pd.DataFrame(data=data, columns=columns, index=index)
        return df.sort_values(by=columns_to_sort).values

    def get_best_model(self, r2s_and_models):
        r2_list_model_list = list(zip(*r2s_and_models))
        r2s = r2_list_model_list[0]
        models = r2_list_model_list[1]
        index = r2s.index(max(r2s))
        return models[index]
