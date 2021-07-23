import torch
from .utilities import Broadcastable

def linear_plateau(x: Broadcastable, a: Broadcastable, b: Broadcastable, r: float):
    r_inv = 1 / r
    a1 = - torch.nn.functional.relu(- x + a + r_inv / 2)
    a2 = torch.nn.functional.relu(- x + (a - r_inv / 2))
    a3 = - torch.nn.functional.relu(x - b + r_inv / 2)
    a4 = torch.nn.functional.relu(x - b - r_inv / 2)
    return 1 + r * (a1 + a2 + a3 + a4)