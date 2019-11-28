import numpy as np

class CircuitCrossValidator:

    def __init__(self, n_sections=5):
        self.n_sections = n_sections

    def validate(self, model, epochs, np_u_t, np_u_i, np_f_t, np_f_v):
        sets = self.split(np_u_t, np_u_i, np_f_t, np_f_v)
        print(sets)

    def split(self, np_u_t, np_u_i, np_f_t, np_f_v):
        return list(zip(np.split(np_u_t, self.n_sections),
                        np.split(np_u_i, self.n_sections),
                        np.split(np_f_t, self.n_sections),
                        np.split(np_f_v, self.n_sections)))
