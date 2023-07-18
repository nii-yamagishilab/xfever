# Copyright (c) 2023, Yamagishi Laboratory, National Institute of Informatics
# Authors: Yi-Chen Chang (yichen@nlplab.cc), Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import math
import torch

from torch.nn import functional as F, CosineSimilarity, MSELoss


def KL(logits1, logits2, num_labels=3):

    prob1_log = F.log_softmax(logits1, dim=-1)
    prob2 = F.softmax(logits2, dim=-1)

    return F.kl_div(prob1_log, prob2, reduction="batchmean")


def KLDiv_detach(logits1, logits2, num_labels=3):

    return KL(
        logits2.view(-1, num_labels),
        logits1.view(-1, num_labels).detach(),
    )


def symmetric_KLDiv(logits1, logits2, num_labels=3):

    consistency_loss_f = KL(
        logits2.view(-1, num_labels),
        logits1.view(-1, num_labels),
    )
    consistency_loss_b = KL(
        logits1.view(-1, num_labels),
        logits2.view(-1, num_labels),
    )
    loss_consistency = consistency_loss_b + consistency_loss_f

    return loss_consistency


def symmetric_KLDiv_detach(logits1, logits2, num_labels=3):

    consistency_loss_f = KL(
        logits2.view(-1, num_labels),
        logits1.view(-1, num_labels).detach(),
    )
    consistency_loss_b = KL(
        logits1.view(-1, num_labels),
        logits2.view(-1, num_labels).detach(),
    )
    loss_consistency = consistency_loss_b + consistency_loss_f

    return loss_consistency


def symmetric_cross_entropy(logits1, logits2, num_labels=3):

    consistency_loss_f = F.cross_entropy(
        logits2.view(-1, num_labels),
        F.softmax(logits1.view(-1, num_labels), dim=-1),
    )
    consistency_loss_b = F.cross_entropy(
        logits1.view(-1, num_labels),
        F.softmax(logits2.view(-1, num_labels), dim=-1),
    )
    loss_consistency = consistency_loss_b + consistency_loss_f

    return loss_consistency


def JSDiv(logits1, logits2, num_labels=3):

    prob1 = F.softmax(logits1, dim=-1)
    prob2 = F.softmax(logits2, dim=-1)

    M = 0.5 * (prob1 + prob2)

    KL_PM = KL(
        M.view(-1, num_labels),
        logits1.view(-1, num_labels),
    )
    KL_QM = KL(
        M.view(-1, num_labels),
        logits2.view(-1, num_labels),
    )

    return 0.5 * (KL_PM + KL_QM)


def JSDis(logits1, logits2, num_labels=3):

    JSDiv_score = JSDiv(logits1, logits2, num_labels)

    return math.sqrt(JSDiv_score)


def CosSimilarityLoss(inputs1, inputs2):

    cos = CosineSimilarity()
    cos_sim = cos(inputs1, inputs2)
    neg_cos_sim = 1 - cos_sim
    neg_cos_sim_loss = torch.mean(neg_cos_sim)

    return neg_cos_sim_loss


def EuclideanDistance(inputs1, inputs2):

    mse_loss_fct = MSELoss(reduction="mean")
    distance = mse_loss_fct(inputs1, inputs2)

    return distance
