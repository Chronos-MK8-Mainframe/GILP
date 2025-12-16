import torch

class PoincareManifold:
    """
    Implements the Poincaré Ball model of hyperbolic geometry.
    Curvature c = 1 (radius 1).
    """
    def __init__(self, eps=1e-5):
        self.eps = eps

    def proj(self, x):
        """
        Project points to be within the Poincaré ball (norm < 1).
        """
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        max_norm = 1.0 - self.eps
        cond = norm > max_norm
        projected = x / norm * max_norm
        return torch.where(cond, projected, x)

    def mobius_add(self, x, y):
        """
        Möbius addition in the Poincaré ball:
        x + y = frac{(1 + 2 <x, y> + ||y||^2) x + (1 - ||x||^2) y}{1 + 2 <x, y> + ||x||^2 ||y||^2}
        
        Optimized implementation from Geoopt/Hyptorch.
        """
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        
        num = (1 + 2 * xy + y2) * x + (1 - x2) * y
        denom = 1 + 2 * xy + x2 * y2
        
        return num / (denom + self.eps)

    def dist(self, x, y):
        """
        Hyperbolic distance:
        d(x, y) = arccosh(1 + 2 ||x-y||^2 / ((1-||x||^2)(1-||y||^2)))
        """
        sq_norm_diff = torch.sum((x - y) ** 2, dim=-1, keepdim=True)
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        
        # Stability: 1 - ||x||^2 can be small
        denom = (1 - x2) * (1 - y2)
        denom = torch.clamp(denom, min=self.eps)
        
        arg = 1 + 2 * sq_norm_diff / denom
        arg = torch.clamp(arg, min=1.0 + self.eps)
        
        return torch.acosh(arg).squeeze(-1)

    def exp_map0(self, v):
        """
        Exponential map at the origin (Euclidean -> Hyperbolic).
        Input: v (vector in tangent space at 0).
        Output: x (point in Poincaré ball).
        Formula: x = tanh(||v||) * (v / ||v||)
        """
        norm = torch.norm(v, p=2, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=self.eps)
        scale = torch.tanh(norm) / norm
        return v * scale

    def log_map0(self, x):
        """
        Logarithmic map at the origin (Hyperbolic -> Euclidean).
        Input: x (point in Poincaré ball).
        Output: v (vector in tangent space at 0).
        Formula: v = arctanh(||x||) * (x / ||x||)
        """
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=self.eps, max=1.0 - self.eps)
        scale = torch.atanh(norm) / norm
        return x * scale
