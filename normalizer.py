class Normalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def normalize(self, np_data):
        if self.mean is None:
            self.mean = np_data.mean()
        if self.std is None:
            self.std = np_data.std()
        return (np_data - self.mean)/self.std

    def denormalize(self, np_data):
        if self.mean is None or self.std is None:
            raise Exception('Undefined mean and std params for denormalization')
        else:
            return np_data*self.std + self.mean
