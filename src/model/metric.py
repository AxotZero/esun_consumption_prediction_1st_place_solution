import torch
import numpy as np
import torch.nn.functional as F

from constant import target_indices

def rmse(output, target):
    with torch.no_grad():
        output *= 100
        target *= 100
        mse = F.mse_loss(output, target)
        rmse = torch.sqrt(mse).item()
    return rmse


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


logs = [np.log2(2), np.log2(3), np.log(4)]
def NDCG(output, target):
    with torch.no_grad():
        _, output_topk_indices = torch.topk(output, 3, dim=1)
        output_topk = torch.gather(target, 1, output_topk_indices)
        target_topk, _ = torch.topk(target, 3, dim=1)

        for i in range(3):
            output_topk[:, i] = output_topk[:, i]/logs[i]
            target_topk[:, i] = target_topk[:, i] / logs[i]
        dcg = torch.sum(output_topk, dim=1)
        idcg = torch.sum(target_topk, dim=1)
        ndcg = dcg/idcg
        ndcg = ndcg[~ndcg.isnan()]
        ndcg = torch.mean(ndcg)
    return ndcg


def NDCG16(output, target):
    return NDCG(output[:, target_indices], target[:, target_indices])




