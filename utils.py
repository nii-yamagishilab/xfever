# Copyright (c) 2023, Yamagishi Laboratory, National Institute of Informatics
# Authors: Yi-Chen Chang (yichen@nlplab.cc), Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import torch

from torch.nn import functional as F


def kl(logits1, logits2, num_labels=3):
    prob1_log = F.log_softmax(logits1, dim=-1)
    prob2 = F.softmax(logits2, dim=-1)
    return F.kl_div(prob1_log, prob2, reduction="batchmean")


def j(logits1, logits2, num_labels=3):
    return kl(logits2.view(-1, num_labels), logits1.view(-1, num_labels),) + kl(
        logits1.view(-1, num_labels),
        logits2.view(-1, num_labels),
    )


def js(logits1, logits2, num_labels=3):
    prob1 = F.softmax(logits1, dim=-1)
    prob2 = F.softmax(logits2, dim=-1)
    m = 0.5 * (prob1 + prob2)
    kl_pm = kl(
        m.view(-1, num_labels),
        logits1.view(-1, num_labels),
    )
    kl_qm = kl(
        m.view(-1, num_labels),
        logits2.view(-1, num_labels),
    )
    return 0.5 * (kl_pm + kl_qm)


def mse(inputs1, inputs2):
    return F.mse_loss(inputs1, inputs2)


def cos(inputs1, inputs2):
    return torch.mean(1.0 - F.cosine_similarity(inputs1, inputs2))
