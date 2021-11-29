from pdb import set_trace as bp
import torch.nn.functional as F
import torch

from constant import target_indices


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


def seq2seq_drop_zero(output, target):
    output = output[:, :-1]
    # set_trace()
    output = output.contiguous().view(-1, int(output.size()[-1]))
    target = target.contiguous().view(-1, int(target.size()[-1]))
    # drop all zero target
    indices = (target.sum(dim=1) != 0).type(torch.bool)
    target = target[indices]
    output = output[indices]
    return output, target


def seq2seq_soft_ce(output, target):
    output, target = seq2seq_drop_zero(output, target)
    return soft_cross_entropy_loss(output, target)


def seq2seq_soft_ce16(output, target):
    output = output[:, :, target_indices]
    target = target[:, :, target_indices]
    output, target = seq2seq_drop_zero(output, target)
    target = target / torch.sum(target, dim=1, keepdim=True)
    return soft_cross_entropy_loss(output, target)


def seq2seq_mse(output, target):
    output = output[:, :-1] # first 23 months
    return F.mse_loss(output, target)