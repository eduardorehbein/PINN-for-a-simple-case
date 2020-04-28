class Normalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def parametrize(self, np_data):
        self.mean = np_data.mean()
        self.std = np_data.std()

    def normalize(self, np_data):
        if self.mean is None or self.std is None:
            self.parametrize(np_data)
        return (np_data - self.mean)/self.std

    def denormalize(self, np_data):
        if self.mean is None or self.std is None:
            raise Exception('Undefined params for denormalization, the normalizer need to be parametrized.')
        else:
            return np_data*self.std + self.mean
