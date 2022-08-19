import torch

import torch.nn as nn

import numpy as np
from .noised_topk import _noised_topk
from .functional import bal_noised_topk_loss, imbal_noised_topk_loss


class _ABCNoisedTopK(nn.Module):
    """
    Abstract Base Class for building a noised top-k loss

    Parameters
    ----------
    k : integer
        Constant to optimize top-k accuracy

    epsilon : float
        Smoothing constant for the top-k function

    n_sample : int
        Number of random normal vectors drawn for each evaluation of the loss

    References
    ----------
    .. [1] C. Garcin, M. Servajean, A.Joly, J. Salmon
      "Stochastic smoothing of the top-K calibrated hinge loss for deep imbalanced classification", ICML 2022,
      https://proceedings.mlr.press/v162/garcin22a/garcin22a.pdf
    """
    def __init__(self, k, epsilon, n_sample):
        super().__init__()
        self.k = k
        self.n_sample = n_sample
        self.epsilon = epsilon
        self._noised_topk = _noised_topk


class BalNoisedTopK(_ABCNoisedTopK):
    """
    Balanced top-k loss as described in [1]

    Parameters
    ----------
    k : integer
        Constant to optimize top-k accuracy

    epsilon : float
        Smoothing constant for the top-k function

    n_sample : int, optional
        Number of random normal vectors drawn for each evaluation of the loss. Defaults to 5.


    Examples
    --------
    import torch
    from noised_topk import BalNoisedTopK

    criteria = BalNoisedTopK(k=2, epsilon=1.0)
    scores = torch.tensor([[2.0, 1.5, -3.0],
                           [7.5, 4.0, -1.5]])
    labels = torch.tensor([0, 2])
    loss_batch = criteria(scores, labels)


    References
    ----------
    .. [1] C. Garcin, M. Servajean, A.Joly, J. Salmon
      "Stochastic smoothing of the top-K calibrated hinge loss for deep imbalanced classification", ICML 2022,
      https://proceedings.mlr.press/v162/garcin22a/garcin22a.pdf
    """
    def __init__(self, k, epsilon, n_sample=5):
        super(BalNoisedTopK, self).__init__(k=k, epsilon=epsilon, n_sample=n_sample)
        self.bal_noised_topk_loss = bal_noised_topk_loss

    def forward(self, s, y):
        d = s.size()[-1]
        n_batch = s.size()[0]
        Z = torch.randn(n_batch, d, self.n_sample, device=s.device)
        return self.bal_noised_topk_loss(s, y, self.k, self.epsilon, self._noised_topk, Z)


class ImbalNoisedTopK(_ABCNoisedTopK):
    """
    Imbalanced top-k loss as described in [1]

    Parameters
    ----------
    k : integer
        Constant to optimize top-k accuracy

    epsilon : float
        Smoothing constant for the top-k function

    max_m : float
        Value of the maximum margin (corresponds to the margin of the class with the smallest number of training examples)

    cls_num_list : list of ints
        List of the number of training examples for each class. The first element corresponds to the first class, etc...

    scale : int, optional
        Scaling parameter after normalization of last layer and last input vector. Defaults to 50. See [1] for more details.
        CAUTION ! This assumes that normalization is performed. If not the case, set scale to 1.

    n_sample : int, optional
        Number of random normal vectors drawn for each evaluation of the loss. Defaults to 5.


    Examples
    --------
    import torch
    from noised_topk import ImbalNoisedTopK

    criteria = ImbalNoisedTopK(k=2, epsilon=0.1, max_m=0.3, cls_num_list=[5, 15, 10])
    scores = torch.tensor([[2.0, 1.5, -3.0],
                           [7.5, 4.0, -1.5]])
    labels = torch.tensor([0, 2])
    loss_batch = criteria(scores, labels)


    References
    ----------
    .. [1] C. Garcin, M. Servajean, A.Joly, J. Salmon
      "Stochastic smoothing of the top-K calibrated hinge loss for deep imbalanced classification", ICML 2022,
      https://proceedings.mlr.press/v162/garcin22a/garcin22a.pdf
    """
    def __init__(self, k, epsilon, max_m, cls_num_list, scale=50, n_sample=5):
        super(ImbalNoisedTopK, self).__init__(k=k, epsilon=epsilon, n_sample=n_sample)
        m_list = 1.0 / np.power(cls_num_list, 0.25)
        m_list = m_list * (max_m / np.max(m_list))
        m_list = m_list.astype(np.float32)
        self.register_buffer("m_list", torch.from_numpy(m_list))
        self.imbal_noised_topk_loss = imbal_noised_topk_loss
        self.scale = scale

    def forward(self, s, y):
        d = s.size()[-1]
        n_batch = s.size()[0]
        Z = torch.randn(n_batch, d, self.n_sample, device=s.device)
        return self.imbal_noised_topk_loss(s, y, self.k, self.m_list, self.epsilon, self._noised_topk, Z, self.scale)