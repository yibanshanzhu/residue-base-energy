from __future__ import annotations

import torch
import torch.nn.functional as F


def _normalize_pwm_tensor(pwm: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pwm = torch.clamp(pwm, min=eps)
    return pwm / pwm.sum(dim=-1, keepdim=True)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=value.dtype, device=value.device)
    denom = mask.sum().clamp_min(1.0)
    return (value * mask).sum() / denom


def compute_rbe_losses(outputs: dict, sample: dict, weights: dict) -> dict:
    pwm_target = _normalize_pwm_tensor(sample["pwm_target"].float())
    log_pwm_pred = F.log_softmax(outputs["pwm_logits"], dim=-1)
    loss_pwm = F.kl_div(log_pwm_pred, pwm_target, reduction="batchmean")

    A_base_label = sample["A_base_label"].float()
    pwm_mask = sample.get(
        "pwm_mask", torch.ones(A_base_label.shape[1], device=A_base_label.device)
    ).float()
    structure_mask = pwm_mask.unsqueeze(0).expand_as(A_base_label)
    A_base_mask = (
        sample.get("A_base_mask", torch.ones_like(A_base_label)).float()
        * structure_mask
    )
    A_backbone_label = sample["A_backbone_label"].float()
    observed_positive_slot = (A_base_label * A_base_mask).sum(dim=0, keepdim=True) > 0.0
    teacher_gate_with_unknown = torch.where(
        A_base_mask > 0.0,
        A_base_label,
        outputs["A_base"].detach(),
    )
    teacher_gate = torch.where(
        observed_positive_slot,
        teacher_gate_with_unknown,
        outputs["A_base"].detach(),
    )
    teacher_pwm_logits = outputs["pwm_prior_logits"] + torch.sum(
        teacher_gate.unsqueeze(-1) * outputs["E"], dim=0
    )
    loss_pwm_teacher = F.kl_div(
        F.log_softmax(teacher_pwm_logits, dim=-1),
        pwm_target,
        reduction="batchmean",
    )

    noncontact_contrib = (
        (1.0 - A_base_label).unsqueeze(-1)
        * A_base_mask.unsqueeze(-1)
        * outputs["A_base"].unsqueeze(-1)
        * outputs["E"]
    )
    loss_noncontact = _masked_mean(
        noncontact_contrib.abs(),
        A_base_mask.unsqueeze(-1).expand_as(noncontact_contrib),
    )

    loss_A_base = _masked_mean(
        F.binary_cross_entropy_with_logits(
            outputs["A_base_logits"], A_base_label, reduction="none"
        ),
        A_base_mask,
    )
    loss_A_backbone = _masked_mean(
        F.binary_cross_entropy_with_logits(
            outputs["A_backbone_logits"], A_backbone_label, reduction="none"
        ),
        structure_mask,
    )
    loss_site = F.binary_cross_entropy_with_logits(
        outputs["site_score"], sample["site_label"].float()
    )
    loss_sparse = _masked_mean(outputs["A_contact"], structure_mask)
    loss_A = loss_A_base + loss_A_backbone

    total = (
        loss_pwm
        + float(weights.get("lambda_pwm_teacher", 1.0)) * loss_pwm_teacher
        + float(weights.get("lambda_A_base", weights.get("lambda_A", 1.0)))
        * loss_A_base
        + float(weights.get("lambda_A_backbone", weights.get("lambda_A", 1.0)))
        * loss_A_backbone
        + float(weights["lambda_site"]) * loss_site
        + float(weights["lambda_sparse"]) * loss_sparse
        + float(weights.get("lambda_noncontact", 0.05)) * loss_noncontact
    )
    return {
        "loss": total,
        "loss_pwm": loss_pwm,
        "loss_pwm_teacher": loss_pwm_teacher,
        "loss_A": loss_A,
        "loss_A_base": loss_A_base,
        "loss_A_backbone": loss_A_backbone,
        "loss_site": loss_site,
        "loss_sparse": loss_sparse,
        "loss_noncontact": loss_noncontact,
    }
