import torch

def linear_plateau(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, r: float):
    # Types check
    assert isinstance(x, torch.Tensor)
    assert isinstance(a, torch.Tensor)
    assert isinstance(b, torch.Tensor)
    assert isinstance(r, (int, float))
    
    # Dimensionality check: x: (1, N); a, b: (P, 1)
    assert x.dim() == a.dim() == b.dim() == 2
    assert x.shape[0] == 1
    assert a.shape[1] == b.shape[1] == 1
    
    r_inv = 1 / r
    a1 = - torch.nn.functional.relu(- x + a + r_inv / 2)
    a2 = torch.nn.functional.relu(- x + (a - r_inv / 2))
    a3 = - torch.nn.functional.relu(x - b + r_inv / 2)
    a4 = torch.nn.functional.relu(x - b - r_inv / 2)
    return 1 + r * (a1 + a2 + a3 + a4)