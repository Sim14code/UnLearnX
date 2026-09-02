import torch
import torch.nn.functional as F


def compute_kl_loss(
    original_logits,
    new_logits
):
    original_probs = F.softmax(
        original_logits,
        dim=-1
    )

    new_log_probs = F.log_softmax(
        new_logits,
        dim=-1
    )

    kl_loss = F.kl_div(
        new_log_probs,
        original_probs,
        reduction="batchmean"
    )

    return kl_loss