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

    A_base_label = sample["A_base_label"].float()
    A_backbone_label = sample["A_backbone_label"].float()
    A_contact_label = sample["A_contact_label"].float()
    teacher_gate = A_base_label
    empty_slot = teacher_gate.sum(dim=0, keepdim=True) <= 0.0
    teacher_gate = torch.where(empty_slot, outputs["A_base"].detach(), teacher_gate)
    teacher_pwm_logits = torch.sum(teacher_gate.unsqueeze(-1) * outputs["E"], dim=0)
    loss_pwm_teacher = F.kl_div(
        F.log_softmax(teacher_pwm_logits, dim=-1),
        pwm_target,
        reduction="batchmean",
    )

    noncontact_contrib = (
        (1.0 - A_base_label).unsqueeze(-1)
        * outputs["A_base"].unsqueeze(-1)
        * outputs["E"]
    )
    loss_noncontact = noncontact_contrib.abs().mean()

    loss_A_base = F.binary_cross_entropy_with_logits(
        outputs["A_base_logits"], A_base_label
    )
    loss_A_backbone = F.binary_cross_entropy_with_logits(
        outputs["A_backbone_logits"], A_backbone_label
    )
    loss_site = F.binary_cross_entropy_with_logits(
        outputs["site_score"], sample["site_label"].float()
    )
    loss_sparse = outputs["A_contact"].mean()
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
