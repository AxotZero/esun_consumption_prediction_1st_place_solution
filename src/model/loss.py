from pdb import set_trace as bp
import torch.nn.functional as F
import torch

from constant import target_indices, logs


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


# def weighted_soft_ce(output, target):
#     mul = torch.mul(-F.log_softmax(output, dim=1), target)
#     _, target_topk_indices = torch.topk(target, 3, dim=1)
#     weights = torch.full(target.size(), 0.5).to(output.get_device())
#     weights[]
#     pass


def label_smoothing_ce(output, target, epsilon=0.2):
    n = output.size()[-1]
    log_preds = F.log_softmax(output, dim=-1)
    loss = -log_preds.sum(dim=-1).mean()
    nll = torch.sum(torch.mul(-log_preds, target)) / output.shape[0]
    return epsilon * (loss/n) + (1-epsilon) * nll



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

    [0.01, 0.2, 0.09, 0.4, 0.3]


def seq2seq_soft_ce16(output, target):
    output = output[:, :, target_indices]
    target = target[:, :, target_indices]
    output, target = seq2seq_drop_zero(output, target)
    target = target / torch.sum(target, dim=1, keepdim=True)
    return soft_cross_entropy_loss(output, target)


def seq2seq_label_smooth_ce(output, target):
    output, target = seq2seq_drop_zero(output, target)
    return label_smoothing_ce(output, target)


def seq2seq_label_smooth_ce16(output, target):
    output = output[:, :, target_indices]
    target = target[:, :, target_indices]
    output, target = seq2seq_drop_zero(output, target)
    target = target / torch.sum(target, dim=1, keepdim=True)
    return label_smoothing_ce(output, target)


def seq2seq_soft_ce16_top3(output, target):
    device = target.get_device()
    output = output[:, :, target_indices]
    target = target[:, :, target_indices]
    output, target = seq2seq_drop_zero(output, target)

    n_rows, n_classes = target.size()
    _, target_top3_indices = torch.topk(target, 3, dim=1)
    flatten_indices = target_top3_indices + (torch.arange(n_rows).view(-1, 1) * n_classes).to(device)
    new_target = torch.zeros(n_rows * n_classes).to(device)
    new_target[flatten_indices] = target.view(-1)[flatten_indices]
    target = new_target.view(n_rows, n_classes)
    target = target / torch.sum(target, dim=1, keepdim=True)
    return soft_cross_entropy_loss(output, target)