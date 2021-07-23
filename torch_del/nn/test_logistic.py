import torch
import numpy as np
from . import Logistic


def test_logistic():
    m = 0
    s = 1
    a, b = -10, 10

    x = torch.linspace(a, b, 101)
    out = Logistic.density(x, m, s)
    integral = out.sum() * (b - a) / len(x)
    assert tuple(out.shape) == (len(x), )
    # The integral must be close to one
    assert abs(1 - integral) < 0.02

    out = Logistic.cumulative(x, m, s)
    assert tuple(out.shape) == (len(x), )
    # Cumulative must be monotonically increasing
    assert min(out[1:] - out[:-1]) >= 0
    # Its minimum and maximum are 0 and 1
    assert 0 <= min(out) < 0.001
    assert 0.999 < max(out) <= 1

    p = torch.linspace(0, 1, 101)
    out = Logistic.quantile(p, m, s)
    assert tuple(out.shape) == (len(p), )
    # Quantile function must be monotonically increasing
    assert min(out[1:] - out[:-1]) >= 0
    # It diverges at -inf and +inf in 0 and 1
    assert out[0] == -np.inf
    assert out[-1] == np.inf
