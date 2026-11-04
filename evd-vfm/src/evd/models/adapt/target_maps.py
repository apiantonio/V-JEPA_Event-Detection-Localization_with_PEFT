"""``TargetModuleMap``: the antidote to the fused QKV.

On V-JEPA 2.1 the three attention projections are a single ``nn.Linear(dim, 3*dim)``
named ``qkv`` (``app/vjepa_2_1/models/utils/modules.py:149``, layout ``[q|k|v]``).
Hand-written ``q_proj``/``k_proj``/``v_proj`` patterns would match nothing and train
nothing, so:

* every backbone declares its map (no hard-coded strings in configs);
* the map declares the slice order instead of assuming it;
* ``resolve`` returns references with the output slice, which makes ``qk_only``
  expressible on a fused QKV;
* a test counts the matched modules on the real model for every placement.

On a fused QKV a rank-*r* adapter on the whole matrix is not equivalent to three
rank-*r* adapters (``fused-r ⊊ separate-r ⊊ fused-3r``), which is why the budget
parity is expressed in trainable parameters, not rank.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

Role = Literal["q", "k", "v", "o", "mlp"]

_BLOCK_RE = re.compile(r"(?:^|\.)blocks\.(\d+)\.")


@dataclass(frozen=True)
class OutSlice:
    """Slice over the output rows of a ``Linear``.

    The builtin ``slice`` is not hashable on Python 3.10/3.11, and these objects end up in
    frozen dataclasses and cache keys.
    """

    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.stop <= self.start:
            raise ValueError(f"empty or inverted slice: [{self.start}, {self.stop})")

    def as_slice(self) -> slice:
        return slice(self.start, self.stop)

    @property
    def width(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class TargetRef:
    """A resolved target: qualified name, role and optional slice.

    The ``role`` is used by tests (counting targets per projection) and by the parameter
    budget formula.
    """

    name: str
    role: Role
    out_slice: OutSlice | None = None

    @property
    def block(self) -> int | None:
        m = _BLOCK_RE.search(self.name)
        return int(m.group(1)) if m else None


@dataclass(frozen=True)
class Placement:
    """Where adaptation is applied.

    ``kind``:
      ``qkvo``           the four attention projections
      ``attn_mlp``       attention + MLP (fc1, fc2)
      ``qkv``            Q, K and V only
      ``qv``             Q and V only
      ``qk_only``        Q and K only (interaction with RoPE)
      ``last_n_blocks``  like ``qkvo``, only in the last ``last_n`` blocks

    ``last_n`` combines with any ``kind``. ``blocks`` selects explicit zero-based block
    indices instead of ``last_n``. Config syntax: ``"qkvo"``, ``"attn_mlp:4"``,
    ``"last_n_blocks:6"``, ``"attn_mlp@5,11,17,23"`` (the hierarchical layers of
    V-JEPA 2.1-L).
    """

    kind: Literal["qkvo", "attn_mlp", "qk_only", "qkv", "qv", "last_n_blocks"]
    last_n: int | None = None
    blocks: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"qkvo", "attn_mlp", "qk_only", "qkv", "qv", "last_n_blocks"}:
            raise ValueError(f"unknown placement: {self.kind!r}")
        if self.blocks is not None:
            if not self.blocks or any(type(i) is not int or i < 0 for i in self.blocks):
                raise ValueError("blocks needs a non-empty list of non-negative integer indices")
            if len(set(self.blocks)) != len(self.blocks):
                raise ValueError("blocks has duplicate indices")
            if self.last_n is not None or self.kind == "last_n_blocks":
                raise ValueError("explicit blocks and last_n are alternatives")
        if self.kind == "last_n_blocks" and not self.last_n:
            raise ValueError("placement 'last_n_blocks' requires last_n (e.g. 'last_n_blocks:4')")
        if self.last_n is not None and self.last_n <= 0:
            raise ValueError(f"last_n must be > 0, not {self.last_n}")

    @classmethod
    def parse(cls, text: str | Placement) -> Placement:
        if isinstance(text, Placement):
            return text
        target, sep, indices = text.partition("@")
        if sep and not indices.strip():
            raise ValueError("a list of block indices must follow @")
        blocks = tuple(int(i.strip()) for i in indices.split(",")) if sep else None
        kind, _, n = target.partition(":")
        return cls(  # type: ignore[arg-type]
            kind=kind.strip(), last_n=int(n) if n.strip() else None, blocks=blocks
        )

    @property
    def roles(self) -> tuple[Role, ...]:
        if self.kind in ("qkvo", "last_n_blocks"):
            return ("q", "k", "v", "o")
        if self.kind == "attn_mlp":
            return ("q", "k", "v", "o", "mlp")
        if self.kind == "qk_only":
            return ("q", "k")
        if self.kind == "qkv":
            return ("q", "k", "v")
        if self.kind == "qv":
            return ("q", "v")
        raise ValueError(f"unknown placement: {self.kind!r}")

    def __str__(self) -> str:
        target = self.kind if self.last_n is None else f"{self.kind}:{self.last_n}"
        if self.blocks is not None:
            target += "@" + ",".join(str(i) for i in self.blocks)
        return target


@dataclass(frozen=True)
class TargetModuleMap:
    """Target modules of one backbone, by role.

    A non-empty ``fused_qkv`` means separate ``q``/``k``/``v`` do not exist and the three
    projections are slices of the fused module. ``fused_out_dim`` is the width of one
    projection and ``fused_order`` declares the slice order.
    """

    q: tuple[str, ...] = ()
    k: tuple[str, ...] = ()
    v: tuple[str, ...] = ()
    o: tuple[str, ...] = ()
    mlp: tuple[str, ...] = ()
    fused_qkv: tuple[str, ...] = ()
    fused_out_dim: int = 0
    fused_order: tuple[str, ...] = field(default=("q", "k", "v"))

    def __post_init__(self) -> None:
        if self.fused_qkv:
            if self.q or self.k or self.v:
                raise ValueError(
                    "inconsistent map: with fused_qkv separate q/k/v do not exist "
                    "(on V-JEPA 2.1 they are a single Linear(dim, 3*dim))"
                )
            if self.fused_out_dim <= 0:
                raise ValueError("fused_qkv requires fused_out_dim > 0")
            if tuple(sorted(self.fused_order)) != ("k", "q", "v"):
                raise ValueError(f"fused_order must be a permutation of q,k,v: {self.fused_order}")

    @property
    def is_fused(self) -> bool:
        return bool(self.fused_qkv)

    @property
    def n_blocks(self) -> int:
        idx = [int(m.group(1)) for name in self.all_names() for m in [_BLOCK_RE.search(name)] if m]
        return max(idx) + 1 if idx else 0

    def all_names(self) -> tuple[str, ...]:
        return self.q + self.k + self.v + self.o + self.mlp + self.fused_qkv

    def _slice_for(self, role: Role) -> OutSlice:
        pos = self.fused_order.index(role)
        d = self.fused_out_dim
        return OutSlice(pos * d, (pos + 1) * d)

    def resolve(self, placement: str | Placement) -> tuple[TargetRef, ...]:
        """Return the concrete targets of a placement, in block order.

        :raises ValueError: if a required role has no targets, or the placement resolves to
            zero modules (a training that trains nothing).
        """
        p = Placement.parse(placement)
        refs: list[TargetRef] = []
        for role in p.roles:
            if role in ("q", "k", "v") and self.is_fused:
                for name in self.fused_qkv:
                    refs.append(TargetRef(name, role, self._slice_for(role)))
                continue
            names: tuple[str, ...] = getattr(self, role)
            if not names:
                raise ValueError(
                    f"placement '{p}' requires projection '{role}', which this map "
                    f"does not declare (available modules: "
                    f"{ {r: len(getattr(self, r)) for r in ('q', 'k', 'v', 'o', 'mlp')} }, "
                    f"fused_qkv={len(self.fused_qkv)})"
                )
            refs.extend(TargetRef(n, role) for n in names)

        if p.blocks is not None:
            available = {r.block for r in refs if r.block is not None}
            missing = set(p.blocks) - available
            if missing:
                raise ValueError(f"requested blocks are missing: {sorted(missing)}")
            refs = [r for r in refs if r.block in p.blocks]

        if p.last_n is not None:
            n_blocks = self.n_blocks
            if n_blocks == 0:
                raise ValueError(
                    "last_n requires names with a block index (...blocks.N...): this map "
                    "has none, so 'last N blocks' is undefined"
                )
            keep = set(range(max(0, n_blocks - p.last_n), n_blocks))
            refs = [r for r in refs if r.block in keep]
            if not refs:
                raise ValueError(f"placement '{p}' resolves to zero modules over {n_blocks} blocks")

        # Stable order: by block, then by role, so logs and parameter sequences are reproducible.
        order = {r: i for i, r in enumerate(("q", "k", "v", "o", "mlp"))}
        refs.sort(key=lambda r: (r.block if r.block is not None else -1, order[r.role], r.name))
        return tuple(refs)
