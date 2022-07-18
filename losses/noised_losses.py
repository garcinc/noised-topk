import torch
from torch.autograd import Function
from torch.distributions.normal import Normal

import torch.nn.functional as F
import torch.nn as nn

import numpy as np


class SoftTopK(Function):
    @staticmethod
    def forward(ctx, x, Z, k, epsilon):
        x = torch.unsqueeze(x, dim=-1)
        noised_x = x + epsilon * Z
        values, indices = torch.topk(noised_x, k=k, dim=1)

        noised_kth_value = values[:, -1, :]
        noised_kth_index = torch.unsqueeze(indices[:, -1, :], dim=1)

        ctx.noised_tensor_size = noised_x.size()
        ctx.save_for_backward(noised_kth_index)
        return torch.mean(noised_kth_value, dim=-1)

    @staticmethod
    def backward(ctx, grad_output):
        indices,  = ctx.saved_tensors
        noised_tensor_size = ctx.noised_tensor_size

        zeros = torch.zeros(noised_tensor_size, device=indices.device)
        zeros.scatter_(1, indices, 1)
        mean_y_star = torch.mean(zeros, dim=-1)
        return mean_y_star * grad_output[:, None], None, None, None


def balanced_noise_topk_loss(s, y, k, epsilon, softtopk, Z):
    correct_scores = torch.squeeze(torch.gather(s, 1, y[:, None]))
    skp1 = softtopk(s, Z, k + 1, epsilon)
    num = F.relu(torch.ones_like(skp1) + skp1 - correct_scores)

    scalar_loss = torch.mean(num)
    return scalar_loss


def imbalanced_noise_topk_loss(s, y, k, m_list, epsilon, softtopk, Z, scale):
    margins_batch = torch.gather(m_list, dim=0, index=y)
    correct_scores = torch.squeeze(torch.gather(s, 1, y[:, None]))
    skp1 = softtopk(s, Z, k + 1, epsilon)
    num = F.relu(scale*(margins_batch + skp1 - correct_scores))

    scalar_loss = torch.mean(num)
    return scalar_loss


class BalancedNoiseTopKLoss(nn.Module):
    def __init__(self, k, n_sample, epsilon):
        super().__init__()
        self.k = k
        self.n_sample = n_sample
        self.epsilon = epsilon
        self.softtopk = SoftTopK.apply
        self.balanced_noise_topk_loss = balanced_noise_topk_loss

    def forward(self, s, y):
        d = s.size()[-1]
        n_batch = s.size()[0]
        Z = torch.randn(n_batch, d, self.n_sample, device=s.device)
        return self.balanced_noise_topk_loss(s, y, self.k, self.epsilon, self.softtopk, Z)


class ImbalancedNoiseTopKLoss(nn.Module):
    def __init__(self, k, cls_num_list, exp, n_sample, epsilon, max_m, scale):
        super(ImbalancedNoiseTopKLoss, self).__init__()
        m_list = 1.0 / np.power(cls_num_list, exp)
        m_list = m_list * (max_m / np.max(m_list))
        self.register_buffer("m_list", torch.from_numpy(m_list))
        self.k = k
        self.n_sample = n_sample
        self.epsilon = epsilon
        self.softtopk = SoftTopK.apply
        self.imbalanced_noise_topk_loss = imbalanced_noise_topk_loss
        self.scale = scale

    def forward(self, s, y):
        d = s.size()[-1]
        n_batch = s.size()[0]
        Z = torch.randn(n_batch, d, self.n_sample, device=s.device)
        return self.imbalanced_noise_topk_loss(s, y, self.k, self.m_list, self.epsilon, self.softtopk, Z, self.scale)
