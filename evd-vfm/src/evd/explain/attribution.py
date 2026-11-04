"""Local attributions to the head logit, without modifying the Meta forward.

The attentive formula uses d(logit)/d(output) of the actual operation:
d(logit)/d(A_hj) = <d(logit)/d(o_h), v_hj>, so A*grad and the A*value contribution are
the same method. Recomputed weights do not pretend to be in the graph.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import Tensor, nn

from evd.explain.attention import FIDELITY_TOLERANCE, _weights_of, cross_attention_module
from evd.types import BackboneOutput


@dataclass(frozen=True)
class Attribution:
    """:param maps: raw signed maps, one grid per method.
    :param logit: binary logit of the actual forward.
    :param fidelity: relative error of the attentive reconstruction.
    """

    maps: dict[str, np.ndarray]
    logit: float
    fidelity: float


def explain_head(
    head: nn.Module, feats: BackboneOutput, *, extra_cams: bool = False
) -> Attribution:
    """Run one forward and one backward of the head only; hooks are removed even on error.

    :param head: binary head in eval mode.
    :param feats: features of the actual backbone, batch one and single level.
    :param extra_cams: include token adaptations of Grad-CAM and LayerCAM, preserving
        historical output keys by default.
    :return: raw attention, grad*activation before/after norm, A*grad and last-layer
        LeGrad.
    :raises ValueError: if batch, grid or number of logits are incompatible.
    :raises RuntimeError: if the reconstruction is not faithful or the attribution is not
        finite.
    """
    if head.training or feats.tokens.shape[0] != 1 or feats.levels:
        raise ValueError("Explanations: use an eval-mode head, batch 1 and a single level")
    x = feats.tokens.detach().requires_grad_(True)
    captured: dict[str, Tensor] = {}
    cross = cross_attention_module(head)

    def norm_hook(module, args, output):
        captured["norm"] = output

    def cross_hook(module, args, output):
        captured["output"] = output
        with torch.no_grad():
            a, rebuilt = _weights_of(module, *args)
            tokens = args[1]
            b, n, d = tokens.shape
            v = module.kv(tokens).reshape(b, n, 2, module.num_heads, d // module.num_heads)
            captured.update(attention=a, rebuilt=rebuilt, value=v[:, :, 1].transpose(1, 2))

    handles = [head.norm.register_forward_hook(norm_hook), cross.register_forward_hook(cross_hook)]
    try:
        with torch.enable_grad():
            logits = head(replace(feats, tokens=x, pooled=None))
            if logits.shape != (1, 1):
                raise ValueError("The explanation requires a single binary logit")
            gx, gn, go = torch.autograd.grad(
                logits.sum(), (x, captured["norm"], captured["output"])
            )
        output = captured["output"].detach()
        fidelity = float(
            (output - captured["rebuilt"]).float().norm() / output.float().norm().clamp_min(1e-12)
        )
        if not np.isfinite(fidelity) or fidelity > FIDELITY_TOLERANCE[output.dtype]:
            raise RuntimeError(f"Unfaithful cross-attention reconstruction: {fidelity}")
        h = cross.num_heads
        gradient_a = torch.einsum(
            "bhqd,bhnd->bhqn",
            go.reshape(1, 1, h, -1).transpose(1, 2).float(),
            captured["value"].float(),
        )
        a = captured["attention"].float()
        raw = {
            "raw_attention": a.mean(1)[:, 0],
            "gradient_activation": (gn.float() * captured["norm"].float()).sum(-1),
            "gradient_activation_prenorm": (gx.float() * x.float()).sum(-1),
            "attention_gradient": (a * gradient_a).sum(1)[:, 0],
            "legrad_last": gradient_a.clamp_min(0).mean(1)[:, 0],
        }
        if extra_cams:
            # Grad-CAM Eq. 1-2 (arxiv.org/abs/1610.02391): pool gradients over
            # all T,H,W tokens. LayerCAM Eq. 6-8 (mmcheng.net/layercam/): local
            # positive gradients; signed features are not clamped beforehand.
            norm = captured["norm"].float()
            raw.update(
                gradcam=(norm * gn.float().mean(1, keepdim=True)).sum(-1).clamp_min(0),
                gradcam_prenorm=(x.float() * gx.float().mean(1, keepdim=True)).sum(-1).clamp_min(0),
                layercam=(norm * gn.float().clamp_min(0)).sum(-1).clamp_min(0),
            )
        maps = {k: v.detach().float().reshape(feats.grid).cpu().numpy() for k, v in raw.items()}
        if any(not np.isfinite(v).all() for v in maps.values()):
            raise RuntimeError("Non-finite attribution: no publishable artifact")
        return Attribution(maps, float(logits.detach().float().item()), fidelity)
    finally:
        for handle in handles:
            handle.remove()


def integrated_gradients(
    forward: Callable[[Tensor], Tensor],
    original: Tensor,
    baseline: Tensor,
    *,
    steps: tuple[int, ...] = (16, 32, 64),
    atol: float = 0.01,
    rtol: float = 0.01,
    batch_size: int = 1,
) -> tuple[Tensor, dict]:
    """Integrated gradients on features, with Gauss-Legendre quadrature and explicit residual.

    The schedule stops at the first number of nodes whose completeness residual is within
    tolerance, so a long schedule costs only where the integral is hard.

    :param forward: the whole head, initial norm included; returns one logit per batch row.
    :param original: original features, batch dimension of size one.
    :param baseline: features of the reference clip.
    :param batch_size: quadrature nodes evaluated in one head call; ``1`` keeps the
        historical node-by-node path, larger values stack the interpolated inputs on the batch
        axis (the head treats rows independently).
    :return: per-feature contributions and convergence receipt.
    :raises ValueError: on mismatching shapes, invalid steps, tolerances or batch size.
    """
    if original.shape != baseline.shape or not steps or any(s < 1 for s in steps):
        raise ValueError("IG: different shapes or invalid number of steps")
    if batch_size < 1 or (batch_size > 1 and original.shape[0] != 1):
        raise ValueError("IG: batch_size must be positive, and batching needs a single input")
    if (
        tuple(sorted(set(steps))) != steps
        or not np.isfinite([atol, rtol]).all()
        or min(atol, rtol) < 0
    ):
        raise ValueError(
            "IG: steps must be strictly increasing and tolerances finite and non-negative"
        )
    original, baseline = original.detach(), baseline.detach()
    delta = original - baseline
    with torch.no_grad():
        difference = float((forward(original) - forward(baseline)).float().item())
    limit = max(atol, rtol * abs(difference))
    for n in steps:
        points, weights = np.polynomial.legendre.leggauss(n)
        total = torch.zeros_like(original, dtype=torch.float32)
        if batch_size == 1:
            for point, weight in zip(points, weights, strict=True):
                with torch.enable_grad():
                    x = (baseline + float((point + 1) / 2) * delta).requires_grad_(True)
                    gradient = torch.autograd.grad(forward(x).sum(), x)[0]
                total += gradient.float() * float(weight / 2)
        else:
            shape = (-1,) + (1,) * (delta.ndim - 1)
            for begin in range(0, n, batch_size):
                alphas = torch.tensor(
                    (points[begin : begin + batch_size] + 1) / 2, device=delta.device
                ).to(delta.dtype)
                nodes = torch.tensor(
                    weights[begin : begin + batch_size] / 2,
                    device=delta.device,
                    dtype=torch.float32,
                )
                with torch.enable_grad():
                    x = (baseline + alphas.view(shape) * delta).requires_grad_(True)
                    gradient = torch.autograd.grad(forward(x).sum(), x)[0]
                total += (gradient.float() * nodes.view(shape)).sum(0, keepdim=True)
        attribution = delta.float() * total
        residual = abs(float(attribution.sum()) - difference)
        if np.isfinite(residual) and residual <= limit:
            break
    return attribution, {
        "steps": n,
        "residual": residual,
        "limit": limit,
        "converged": bool(np.isfinite(residual) and residual <= limit),
        "logit_difference": difference,
    }


@contextmanager
def full_precision() -> Iterator[None]:
    """Run the enclosed computation in true fp32: autocast and TF32 off, restored on exit.

    With the head in bf16, Integrated Gradients stopped at residuals of 0.1-0.5 logit that more
    quadrature nodes did not lower (``exp/localization_ig.py``); TF32 would keep a 10-bit
    mantissa in fp32 matrix products. Tensors created outside keep their dtype: pass fp32
    inputs.
    """
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        with ExitStack() as stack:
            stack.enter_context(torch.autocast("cpu", enabled=False))
            if torch.cuda.is_available():
                stack.enter_context(torch.autocast("cuda", enabled=False))
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


def integrated_gradients_fixed(
    forward: Callable[[Tensor], Tensor],
    original: Tensor,
    baselines: Mapping[str, Tensor],
    *,
    nodes: int,
    batch_size: int,
    atol: float = 0.01,
    rtol: float = 0.01,
) -> dict[str, tuple[Tensor, dict]]:
    """Integrated gradients of several baselines at a fixed node count, in one work queue.

    Every (baseline, node) pair is one row of a single queue cut into head calls of
    ``batch_size`` rows, so the baselines share full batches instead of each ending with a
    half-empty call. Gauss-Legendre nodes are not nested, so the adaptive schedule of
    :func:`integrated_gradients` recomputes every node when it doubles; a count fixed once
    (by the pilot of ``docs/varie/piano_localizzazione_per_classe_20261021.md`` §7.1) does
    not. :func:`integrated_gradients` is kept unchanged for the bit-identical replay of the
    16 October campaign.

    :param forward: the head on a ``[B, N, D]`` batch, returning one logit per row (``[B]``
        or ``[B, 1]``). Rows must be independent, as in the attentive head.
    :param original: ``[1, N, D]`` features of the clip. Computation runs in its dtype: pass
        fp32 and wrap the call in :func:`full_precision` for a full-precision head.
    :param baselines: name to ``[1, N, D]`` features of each reference clip, same dtype and
        device as ``original``.
    :param nodes: Gauss-Legendre nodes per baseline.
    :param batch_size: rows per head call, for both the endpoint logits and the gradients.
    :param atol: absolute tolerance of the completeness residual.
    :param rtol: tolerance relative to the logit difference.
    :return: name to ``(contributions, receipt)``: ``[1, N, D]`` float32 contributions whose sum
        approximates ``forward(original) - forward(baseline)``, and the receipt with ``nodes``,
        ``residual``, ``limit``, ``converged`` and ``logit_difference``.
    :raises ValueError: without baselines, with a batch dimension other than one, mismatching
        shapes, non-positive ``nodes`` or ``batch_size``, or invalid tolerances.
    """
    if not baselines:
        raise ValueError("IG: at least one baseline is required")
    if original.ndim < 2 or original.shape[0] != 1:
        raise ValueError(f"IG: the original needs a batch dimension of one, got {original.shape}")
    if any(b.shape != original.shape for b in baselines.values()):
        raise ValueError("IG: every baseline must have the shape of the original")
    if nodes < 1 or batch_size < 1:
        raise ValueError(f"IG: nodes and batch_size must be positive, got {nodes}, {batch_size}")
    if not np.isfinite([atol, rtol]).all() or min(atol, rtol) < 0:
        raise ValueError("IG: tolerances must be finite and non-negative")
    names = list(baselines)
    original = original.detach()
    stacked = {name: baselines[name].detach() for name in names}
    deltas = {name: original - stacked[name] for name in names}

    def logits(x: Tensor) -> Tensor:
        return forward(x).reshape(len(x), -1)[:, 0]

    with torch.no_grad():
        endpoints = torch.cat([original, *(stacked[n] for n in names)])
        values = torch.cat(
            [logits(endpoints[i : i + batch_size]) for i in range(0, len(endpoints), batch_size)]
        ).float()
    points, weights = np.polynomial.legendre.leggauss(nodes)
    queue = [(name, j) for name in names for j in range(nodes)]
    totals = {name: torch.zeros_like(original, dtype=torch.float32) for name in names}
    for begin in range(0, len(queue), batch_size):
        rows = queue[begin : begin + batch_size]
        with torch.enable_grad():
            x = torch.cat(
                [stacked[n] + float((points[j] + 1) / 2) * deltas[n] for n, j in rows]
            ).requires_grad_(True)
            gradient = torch.autograd.grad(logits(x).sum(), x)[0]
        for row, (name, j) in enumerate(rows):
            totals[name] += gradient[row : row + 1].float() * float(weights[j] / 2)
    out = {}
    for index, name in enumerate(names):
        difference = float(values[0] - values[index + 1])
        attribution = deltas[name].float() * totals[name]
        residual = abs(float(attribution.sum()) - difference)
        limit = max(atol, rtol * abs(difference))
        out[name] = (
            attribution,
            {
                "nodes": nodes,
                "residual": residual,
                "limit": limit,
                "converged": bool(np.isfinite(residual) and residual <= limit),
                "logit_difference": difference,
            },
        )
    return out
