import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt


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

            model.reset_nn()

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


class PlotValidator:

    @staticmethod
    def compare(np_x_axis, np_validation_output, np_prediction, title='Sampled vs predicted output',
                  validation_label='Sampled output', prediction_label='Predicted output'):
        plt.title(title)
        plt.plot(np_x_axis, np_validation_output, label=validation_label)
        plt.plot(np_x_axis, np_prediction, label=prediction_label)
        plt.legend()
        plt.show()

    @staticmethod
    def multicompare(x_axis, validation_outputs, predictions, titles=None, validation_labels=None,
                     prediction_labels=None):
        len_x_axis = len(x_axis)
        len_validation_outputs = len(validation_outputs)
        len_predictions = len(predictions)
        len_validation_labels = len(validation_labels)
        len_prediction_labels = len(prediction_labels)

        if len_x_axis > 9 or len_validation_outputs > 9 or len_predictions > 9:
            raise Exception('This function does not plot more than 9 signals.')
        if len_validation_outputs != len_predictions:
            raise Exception("Validation outputs' and predictions' sizes do not match.")

        rows = 1
        if len_validation_outputs > 6:
            rows = 3
        elif len_validation_outputs >= 3:
            rows = 2

        columns = 1
        if len_validation_outputs >= 5:
            columns = 3
        elif 1 < len_validation_outputs < 5:
            columns = 2

        standard_validation_label = PlotValidator.standard_plot_param_analysis(len_validation_labels)
        standard_prediction_label = PlotValidator.standard_plot_param_analysis(len_prediction_labels)
        standard_x_plot = PlotValidator.standard_plot_param_analysis(len_x_axis)

        for i in range(len_validation_outputs):
            title = PlotValidator.filter_titles_and_labels(i, titles, 'Plot ' + str(i + 1))

            if standard_validation_label:
                validation_label = validation_labels[0]
            else:
                validation_label = PlotValidator.filter_titles_and_labels(i, validation_labels, 'Sampled output')

            if standard_prediction_label:
                prediction_label = prediction_labels[0]
            else:
                prediction_label = PlotValidator.filter_titles_and_labels(i, prediction_labels, 'Predicted output')

            if standard_x_plot:
                x_plot = x_axis[0]
            else:
                x_plot = x_axis[i]

            plt.subplot(rows, columns, i + 1)
            plt.title(title)
            plt.plot(x_plot, validation_outputs[i], label=validation_label)
            plt.plot(x_plot, predictions[i], label=prediction_label)
            plt.legend()

        plt.show()

    @staticmethod
    def standard_plot_param_analysis(len_param):
        if len_param == 1:
            return True
        else:
            return False

    @staticmethod
    def filter_titles_and_labels(index, param_list, standard_text):
        if len(param_list) > index:
            if not param_list[index]:
                return standard_text
            else:
                return param_list[index]
        else:
            return standard_text
