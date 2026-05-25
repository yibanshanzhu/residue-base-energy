from __future__ import annotations

import torch
import torch.nn.functional as F


def _normalize_pwm_tensor(pwm: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pwm = torch.clamp(pwm, min=eps)
    return pwm / pwm.sum(dim=-1, keepdim=True)


def compute_rbe_losses(outputs: dict, sample: dict, weights: dict) -> dict:
    pwm_target = _normalize_pwm_tensor(sample["pwm_target"].float())
    log_pwm_pred = F.log_softmax(outputs["pwm_logits"], dim=-1)
    loss_pwm = F.kl_div(log_pwm_pred, pwm_target, reduction="batchmean")

    loss_A = F.binary_cross_entropy_with_logits(
        outputs["A_logits"], sample["A_label"].float()
    )
    loss_site = F.binary_cross_entropy_with_logits(
        outputs["site_score"], sample["site_label"].float()
    )
    loss_sparse = outputs["A"].mean()

    total = (
        loss_pwm
        + float(weights["lambda_A"]) * loss_A
        + float(weights["lambda_site"]) * loss_site
        + float(weights["lambda_sparse"]) * loss_sparse
    )
    return {
        "loss": total,
        "loss_pwm": loss_pwm,
        "loss_A": loss_A,
        "loss_site": loss_site,
        "loss_sparse": loss_sparse,
    }

