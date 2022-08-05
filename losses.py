import torch
import torch.nn.functional as F


def padded_crossentropy_loss(predictions, len_predictions, targets, len_targets, device):
    batch_ce_loss = torch.tensor(0.0, device=device)
    for i in range(predictions.size(0)):
        ce = F.cross_entropy(
            predictions[i][:len_predictions[i]],
            targets[i][:len_targets[i]],
            reduction="sum", ignore_index=0)
        ce = ce / len_predictions[i].to(device)
        batch_ce_loss += ce

    return batch_ce_loss / predictions.size(0)
