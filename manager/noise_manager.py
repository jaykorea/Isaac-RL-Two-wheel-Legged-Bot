import numpy as np


class AdditiveNoiseManager:
    def __init__(self, noise_type=None):
        self.noise_type = noise_type

    def apply(self, data, **kwargs):
        if self.noise_type == "constant":
            return ConstantNoise(**kwargs).apply(data)
        elif self.noise_type == "uniform":
            return UniformNoise(**kwargs).apply(data)
        elif self.noise_type == "gaussian":
            return GaussianNoise(**kwargs).apply(data)
        else:
            raise ValueError("Unsupported noise type")


class ConstantNoise:
    def __init__(self, noise_value=0.1):
        self.noise_value = noise_value

    def apply(self, data, noise_value=None):
        if noise_value is None:
            noise_value = self.noise_value

        if isinstance(data, (list, np.ndarray)):
            data = np.array(data)
            return data + noise_value
        else:
            return data + noise_value


class UniformNoise:
    def __init__(self, low=-0.1, high=0.1):
        self.low = low
        self.high = high

    def apply(self, data, low=None, high=None):
        if low is None:
            low = self.low
        if high is None:
            high = self.high

        if isinstance(data, (list, np.ndarray)):
            data = np.array(data)
            noise = np.random.uniform(low, high, size=data.shape)
            return data + noise
        else:
            noise = np.random.uniform(low, high)
            return data + noise


class GaussianNoise:
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def apply(self, data, mean=None, std=None):
        if mean is None:
            mean = self.mean
        if std is None:
            std = self.std

        if isinstance(data, (list, np.ndarray)):
            data = np.array(data)
            noise = np.random.normal(mean, std, size=data.shape)
            return data + noise
        else:
            noise = np.random.normal(mean, std)
            return data + noise