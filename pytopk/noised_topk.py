import torch
import torch.nn as nn
from torch.autograd import Function


class _NoisedTopK(Function):
    """
    Smooth top-k function with automatic differentiation
    """
    @staticmethod
    def forward(ctx, x, Z, k, epsilon, dim, dim_offset):
        x = torch.unsqueeze(x, dim=-1)
        noised_x = x + epsilon * Z

        ctx.noised_tensor_size = noised_x.size()
        ctx.dim = dim
        ctx.dim_offset = dim_offset

        noised_kth_value, noised_kth_index = torch.kthvalue(-noised_x, k=k, dim=dim + dim_offset)

        noised_kth_index_exp = torch.unsqueeze(noised_kth_index, dim=dim + dim_offset)

        ctx.save_for_backward(noised_kth_index_exp)
        res = torch.mean(- noised_kth_value, dim=-1)

        return res

    @staticmethod
    def backward(ctx, grad_output):
        indices,  = ctx.saved_tensors
        noised_tensor_size = ctx.noised_tensor_size
        dim = ctx.dim
        dim_offset = ctx.dim_offset

        zeros = torch.zeros(noised_tensor_size, device=indices.device)

        zeros.scatter_(dim + dim_offset, indices, 1)
        mean_y_star = torch.mean(zeros, dim=-1)

        return mean_y_star * torch.unsqueeze(grad_output, dim=dim), None, None, None, None, None


_noised_topk = _NoisedTopK.apply


class NoisedTopK(nn.Module):
    """
    pytorch Module to compute the smooth (differentiable) top-k value of a tensor along the specified dimension, as described in [1].
    Can be plugged into any deep network architecture.

    Parameters
    ----------
    k : integer
        Constant to get the k-th largest value.
        k should be in the integer range [1, n_dim] where n_dim is the size of dimension dim

    epsilon : float, optional
        Smoothing constant for the top-k function.
        Defaults to 1.0. Increasing epsilon gives more smoothing but a more inaccurate top-k.
        Setting epsilon to 0 gives the true top-k.

    n_sample : int, optional
        Number of random normal vectors drawn for each evaluation of the noised top-k. Defaults to 5.

    dim : int, optional
        dimension along which to compute top-k. Defaults to -1.


    Examples
    --------
    import torch
    from noised_topk import NoisedTopK

    smooth_topk = NoisedTopK(k=3, dim=-1)
    x = torch.tensor([[-1.5, 2.0, 0.7, 3.8],
                      [-1.1, -5.4, 0.1, 2.3]], requires_grad=True)
    out = smooth_topk(x)
    print(out)

    >> tensor([ 0.4823, -1.4710], grad_fn=<_NoisedTopKBackward>)

    out.sum().backward()
    print(x.grad)

    >> tensor([[0.0000, 0.4000, 0.6000, 0.0000],
              [0.8000, 0.0000, 0.2000, 0.0000]])


    References
    ----------
    .. [1] C. Garcin, M. Servajean, A.Joly, J. Salmon
      "Stochastic smoothing of the top-K calibrated hinge loss for deep imbalanced classification", ICML 2022,
      https://proceedings.mlr.press/v162/garcin22a/garcin22a.pdf
    """
    def __init__(self, k, epsilon=1.0, n_sample=5, dim=-1):
        super().__init__()
        self.k = k
        self.epsilon = epsilon
        self.n_sample = n_sample
        self.dim = dim
        self._set_offset()

    def _set_offset(self):
        if self.dim < 0:
            self.dim_offset = -1
        else:
            self.dim_offset = 0

    def forward(self, x):
        Z = torch.randn(*x.size(), self.n_sample, device=x.device)
        return _noised_topk(x, Z, self.k, self.epsilon, self.dim, self.dim_offset)


