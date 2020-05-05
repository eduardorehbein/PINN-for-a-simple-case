class Normalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def parametrize(self, data):
        self.mean = data.mean()
        self.std = data.std()

    def normalize(self, data):
        if self.mean is None or self.std is None:
            self.parametrize(data)
        return (data - self.mean) / self.std

    def denormalize(self, data):
        if self.mean is None or self.std is None:
            raise Exception('Undefined params for denormalization, the normalizer need to be parametrized.')
        else:
            return data * self.std + self.mean
