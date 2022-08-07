import numpy as np
import torch
import torch.nn.functional as F


def padded_crossentropy_loss(predictions, len_predictions, targets, len_targets, device):
    batch_ce_loss = torch.tensor(0.0, device=device)
    for i in range(predictions.size(0)):
        single_predicted_unpadded = predictions[i][:len_predictions[i]]
        single_target_unpadded = targets[i][:len_targets[i]]
        ce = F.cross_entropy(
            single_predicted_unpadded,
            single_target_unpadded,
            reduction="mean", ignore_index=0)

        batch_ce_loss += ce

    return batch_ce_loss / predictions.size(0)


def ce_loss_with_ignore(predictions, targets):
    return F.cross_entropy(predictions, targets, reduction="mean", ignore_index=0)

def ce_kl(predictions, targets, mu, logvar, optimizer_step):
    ce_loss = ce_loss_with_ignore(predictions, targets)
    kl_loss, kl_div, kl_weight = kl_divergence(mu, logvar, optimizer_step)

    return ce_loss + kl_loss, (ce_loss, kl_div, kl_weight)

def kl_divergence(mean, log_var, optimizer_step=None):
    kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    kl_weight = 1
    if optimizer_step:
        k, step, x0 = 0.0025, optimizer_step, 2500
        kl_weight = float(1 / (1 + np.exp(-k * (step - x0))))

    kl_loss = kl_div * kl_weight
    return kl_loss, kl_div, kl_weight


def padded_kl_ce_loss(predictions, len_predictions,
                      targets, len_targets,
                      mean, log_variance,
                      optimizer_step, device):
    ce_loss = padded_crossentropy_loss(predictions, len_predictions, targets, len_targets, device)

    kl_loss, kl_div, kl_weight = kl_divergence(mean, log_variance, optimizer_step)
    kl_loss *= 0

    l = 1 if predictions.size(0) == 0 else predictions.size(0)
    loss = ce_loss + (kl_loss / l)
    return loss, (ce_loss, kl_div, kl_weight)


def padded_kl_nll_loss(predictions, len_predictions,
                       targets, len_targets,
                       mean, log_variance,
                       optimizer_step, device):
    # KL Annealing https://arxiv.org/pdf/1511.06349.pdf
    nll_loss = torch.tensor(0.0, device=device)

    for i in range(predictions.size(0)):
        nll = F.nll_loss(
            predictions[i][:len_predictions[i]],
            targets[i][:len_targets[i]],
            reduction="sum",
            ignore_index=0
        )
        nll = nll / len_predictions[i].cuda()
        nll_loss += nll

    # batch_loss = batch_loss / predictions.size(0)

    kl_loss, kl_div, kl_weight = kl_divergence(mean, log_variance, optimizer_step)

    # ELBO Loss
    l = 1 if predictions.size(0) == 0 else predictions.size(0)
    loss = (nll_loss + kl_loss) / l
    return loss, (nll_loss, kl_div, kl_weight)
