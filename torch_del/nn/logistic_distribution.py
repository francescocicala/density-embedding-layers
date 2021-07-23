import torch
from .utilities import Broadcastable

class Logistic:
    def __init__(self):
        super().__init__()

    def __call__(self, x: Broadcastable, m: Broadcastable, s: Broadcastable):
        return self.density(x, m, s)

    @staticmethod
    def density(x: Broadcastable, m: Broadcastable, s: Broadcastable):
        e = torch.exp(- (x - m) / s)
        return e / (s * torch.pow(1 + e, torch.tensor(2)))

    @staticmethod
    def cumulative(x: Broadcastable, m: Broadcastable, s: Broadcastable):
        e = torch.exp(- (x - m) / s)
        return 1 / (1 + e)

    @staticmethod
    def quantile(p: Broadcastable, m: Broadcastable, s: Broadcastable) -> Broadcastable:
        return m + s * torch.log(p / (1 - p))