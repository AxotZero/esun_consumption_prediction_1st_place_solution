import torch.nn.functional as F
import torch


def nll_loss(output, target):
    return F.nll_loss(output, target)


def mse_loss(output, target):
    return F.mse_loss(output, target)


def rmse_loss(output, target):
    eps = 1e-6
    return torch.sqrt(F.mse_loss(output, target) + eps)


def bce_loss(output, target):
    return F.binary_cross_entropy(output, target)


def bce_loss_with_logits(output, target):
    return F.binary_cross_entropy_with_logits(output, target)

def cc_loss_with_logits(output, target):
    return F.cross_entropy(output, target)

def soft_cross_entropy_loss(output, target):
    """
    src: https://blog.csdn.net/Hungryof/article/details/93738717
    """
    return torch.sum(torch.mul(-F.log_softmax(output, dim=1), target)) / output.shape[0]