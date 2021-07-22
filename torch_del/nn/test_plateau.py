import torch
from . import linear_plateau

def test_linear_plateau():
    x = torch.linspace(0, 100, 101).view(1, -1)
    a = torch.linspace(0, 75, 4).view(-1, 1)
    b = torch.linspace(25, 100, 4).view(-1, 1)
    r = 2
    out = linear_plateau(x, a, b, r)
    # Output dimensionality check.
    assert tuple(out.shape) == (4, 101)
    # Check that the area under the plateaus is correct.
    for i in range(len(out)):
        assert out[i].sum() == 25