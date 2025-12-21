import torch

class LorentzManifold:
    """
    Implements the Lorentz (Hyperboloid) model of hyperbolic geometry.
    Curvature c = 1.
    Points x satisfy: <x, x>_L = -1 and x_0 > 0.
    """
    def __init__(self, eps=1e-5):
        self.eps = eps

    def l_inner(self, u, v):
        """
        Lorentz inner product: -u0*v0 + u1*v1 + ... + un*vn
        """
        res = torch.sum(u * v, dim=-1, keepdim=True)
        # Multiply u0*v0 by -2 and add back to sum to get net -u0*v0
        return res - 2 * (u[..., 0:1] * v[..., 0:1])

    def proj(self, x):
        """
        Project points onto the hyperboloid <x, x>_L = -1, x_0 > 0.
        """
        # Ensure x_0 is positive
        x_0 = torch.clamp(x[..., 0:1], min=1.0)
        x_rest = x[..., 1:]
        
        # Current L-norm squared
        norm_tail_sq = torch.sum(x_rest**2, dim=-1, keepdim=True)
        # We need -x_0^2 + norm_tail_sq = -1  => x_0^2 = 1 + norm_tail_sq
        new_x_0 = torch.sqrt(1 + norm_tail_sq)
        
        return torch.cat([new_x_0, x_rest], dim=-1)

    def dist(self, x, y):
        """
        Distance in Lorentz model: d(x, y) = arccosh(-<x, y>_L)
        """
        inner = self.l_inner(x, y)
        arg = -inner
        # Clamp arg >= 1.0 for acosh stability
        arg = torch.clamp(arg, min=1.0 + self.eps)
        return torch.acosh(arg).squeeze(-1)

    def exp_map0(self, v):
        """
        Exponential map at the 'origin' [1, 0, ..., 0] (Euclidean tangent -> Lorentz).
        Input: v (vector in tangent space at [1, 0...], so v_0 must be 0).
        Output: x (point on hyperboloid).
        x = cosh(||v||_L) * [1, 0...] + sinh(||v||_L) * v / ||v||_L
        """
        # Ensure v_0 is 0 for tangent vectors at the origin
        v_0 = torch.zeros_like(v[..., 0:1])
        v_rest = v[..., 1:]
        v_tangent = torch.cat([v_0, v_rest], dim=-1)
        
        norm_v = torch.norm(v_tangent, p=2, dim=-1, keepdim=True)
        norm_v = torch.clamp(norm_v, min=self.eps)
        
        res_0 = torch.cosh(norm_v)
        res_rest = torch.sinh(norm_v) * (v_tangent[..., 1:] / norm_v)
        
        return torch.cat([res_0, res_rest], dim=-1)

    def log_map0(self, x):
        """
        Logarithmic map at the 'origin' [1, 0, ..., 0] (Lorentz -> Euclidean tangent).
        Input: x (point on hyperboloid).
        Output: v (vector in tangent space at [1, 0...], v_0 = 0).
        """
        # Distance from origin [1, 0...]
        origin = torch.zeros_like(x)
        origin[..., 0] = 1.0
        
        dist = self.dist(origin, x).unsqueeze(-1)
        dist_clamped = torch.clamp(dist, min=self.eps)
        
        # Projection of x onto tangent space at origin
        # The tangent space is the plane x_0 = 1? No, at [1,0...], tangent space is x_0 = 0.
        v_0 = torch.zeros_like(x[..., 0:1])
        v_rest = x[..., 1:]
        
        norm_v_rest = torch.norm(v_rest, p=2, dim=-1, keepdim=True)
        norm_v_rest = torch.clamp(norm_v_rest, min=self.eps)
        
        return torch.cat([v_0, v_rest * (dist / norm_v_rest)], dim=-1)
