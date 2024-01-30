import torch
import torch.nn.functional as F


def bal_noised_topk_loss(s, y, k, epsilon, softtopk, Z):
    """
    Computes the balanced loss described in [1] given a batch of score vectors / labels

    Parameters
    ----------
    s : torch.Tensor (float), shape (batch_size, n_classes)
        Batch of score vectors

    y : torch.Tensor (int), shape (batch_size,)
        Batch of true labels.

    k : integer
        Constant to optimize top-k accuracy

    epsilon : float
        Smoothing constant for the top-k function

    softtopk : function to compute the smooth top-k
        typically noised_topk._noised_topk

    Z : torch.Tensor (float), shape (batch_size, n_classes, n_sample)
        Batch of random normal vectors

    Returns
    -------
    scalar_loss : torch.Tensor, size 0 (scalar)
        Average batch loss

    References
    ----------
    .. [1] C. Garcin, M. Servajean, A.Joly, J. Salmon
      "Stochastic smoothing of the top-K calibrated hinge loss for deep imbalanced classification", ICML 2022,
      https://proceedings.mlr.press/v162/garcin22a/garcin22a.pdf
    """
    correct_scores = torch.squeeze(torch.gather(s, 1, y[:, None]))
    skp1 = softtopk(s, Z, k + 1, epsilon, -1, -1)
    num = F.relu(torch.ones_like(skp1) + skp1 - correct_scores)
    scalar_loss = torch.mean(num)
    return scalar_loss


def imbal_noised_topk_loss(s, y, k, m_list, epsilon, softtopk, Z, scale):
    """
    Computes the imbalanced loss described in [1] given a batch of score vectors / labels

    Parameters
    ----------
    s : torch.Tensor (float), shape (batch_size, n_classes)
        Batch of score vectors

    y : torch.Tensor (int), shape (batch_size,)
        Batch of true labels.

    k : integer
        Constant to optimize top-k accuracy

    epsilon : float
        Smoothing constant for the top-k function

    softtopk : function to compute the smooth top-k
        typically noised_topk._noised_topk

    Z : torch.Tensor (float), shape (batch_size, n_classes, n_sample)
        Batch of random normal vectors

    Returns
    -------
    scalar_loss : torch.Tensor, size 0 (scalar)
        Average batch loss

    References
    ----------
    .. [1] C. Garcin, M. Servajean, A.Joly, J. Salmon
      "Stochastic smoothing of the top-K calibrated hinge loss for deep imbalanced classification", ICML 2022,
      https://proceedings.mlr.press/v162/garcin22a/garcin22a.pdf
    """
    m_list = m_list.to(s.device)
    margins_batch = torch.gather(m_list, dim=0, index=y)
    correct_scores = torch.squeeze(torch.gather(s, 1, y[:, None]))
    skp1 = softtopk(s, Z, k + 1, epsilon, -1, -1)
    num = F.relu(scale*(margins_batch + skp1 - correct_scores))
    scalar_loss = torch.mean(num)
    return scalar_loss
