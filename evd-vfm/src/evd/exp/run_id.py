"""Run identity.

``{exp}_{dataset}_{backbone}_{adapt}_{seed}``, e.g. ``e3r_ucf_vjepa21l_lora-r32-qkvomlp_s2``:
a parseable name answers "where does this number come from" by querying the registry.

The ``k`` field left the grammar with ADR-031 (no labelling budget in the fully
supervised regime). Older five-field run ids remain readable: :meth:`RunId.parse` keeps
the value in :attr:`RunId.k_legacy`.

Grammar extensions (none of them approves a protocol by itself):

* ``smoke``: smoke tests write to the registry and must be filterable out;
* ``e0e``-``e0i``: pilots (geometry, augmentation, weight decay, LR selection); seeds
  accept any positive integer;
* ``e3s``: sequential combinations (ADR-065), where one family comes already trained
  from another run (merged with ``fondi`` or frozen with ``congela``); a separate
  ``exp`` because the ``run_id`` has no pre-adaptation field;
* ``e3m`` and ``e3c``: the three-epoch matrix and the reparam+prompt combinations;
  the budget is named in the ``exp`` because the ``run_id`` has no budget field;
* ``e0i2``: the second E0i campaign, whose grid has a different cardinality.

The ``adapt`` field allows only lowercase letters, digits and ``-``: the ``run_id``
becomes a folder name, and ``:`` is illegal in Windows paths.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: ``e1c`` is allowed only if the ADR-017 window opens.
EXPS = frozenset(
    {
        "e0a",
        "e0b",
        "e0c",
        "e0d",
        "e0e",
        "e0f",
        "e0g",
        "e0h",
        "e0i",
        "e0i2",
        "e1c",
        "e2a",
        "e2b",
        "e2c",
        "e2d",
        "e3r",
        "e3g",
        "e3p",
        "e3m",
        "e3m4",
        "e3c",
        "e3c4",
        "e3s",
        "e3sv",
        "e3so",
        "e3x",
        "e3d",
        "e3e",
        "e4a",
        "e4b",
        "e5a",
        "e5b",
        "e5c",
        # Class-routing localization heads, plan of 21 October 2026 (B1, B2, B3-sum, B3-GradNorm).
        "lcb1",
        "lcb2",
        "lcb3s",
        "lcb3g",
        # Class heads alone on a trained V-JEPA 2.1-L encoder, 26 October 2026 (B4: LoRA four
        # epochs; LoRA one epoch then VPT-shallow three epochs).
        "lcb4l",
        "lcb4v",
        "demo",
        "smoke",
    }
)
DATASETS = frozenset({"ucf", "xd", "ucf101mini", "synth"})
BACKBONE_IDS = frozenset(
    {
        "vjepa21b",
        "vjepa21l",
        "vjepa21giant",
        "vjepa21gigantic",
        "ivnl",
        "ivnl-s1",
        "dinov3l",
        "tiny",
    }
)
#: Values of the ``k`` field before ADR-031, kept only to reread existing names.
K_VALUES_LEGACY = frozenset({"1", "2", "4", "8", "16", "32", "full", "na"})

_ADAPT_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_RUN_ID_RE = re.compile(r"^(?P<rest>.+?)(?:_k(?P<k>[a-z0-9]+))?_s(?P<seed>\d+)$")


@dataclass(frozen=True)
class RunId:
    exp: str
    dataset: str
    backbone: str
    adapt: str
    seed: int = 1
    k_legacy: str | None = None
    """Set only when rereading a few-shot ``run_id``; :meth:`format` re-emits it for
    fidelity to the original name."""

    def __post_init__(self) -> None:
        if self.exp not in EXPS:
            raise ValueError(f"exp '{self.exp}' outside the grammar: {sorted(EXPS)}")
        if self.dataset not in DATASETS:
            raise ValueError(f"dataset '{self.dataset}' outside the grammar: {sorted(DATASETS)}")
        if self.backbone not in BACKBONE_IDS:
            raise ValueError(
                f"backbone '{self.backbone}' outside the grammar: {sorted(BACKBONE_IDS)}"
            )
        if not _ADAPT_RE.match(self.adapt):
            raise ValueError(
                f"adapt '{self.adapt}' not allowed: only lowercase letters, digits and '-' "
                f"(the run_id becomes a folder name)"
            )
        if self.k_legacy is not None and self.k_legacy not in K_VALUES_LEGACY:
            raise ValueError(
                f"k '{self.k_legacy}' outside the historical grammar: "
                f"{sorted(K_VALUES_LEGACY)}. The k field is no longer generated (ADR-031): "
                f"it exists only to reread runs older than 27 August 2026"
            )
        if self.seed <= 0:
            raise ValueError(f"seed must be positive, not {self.seed}")

    @property
    def is_legacy(self) -> bool:
        """:return: whether the name comes from the few-shot regime (five fields)."""
        return self.k_legacy is not None

    @property
    def is_official(self) -> bool:
        """Return whether this is an official run (smoke runs are not)."""
        return self.exp != "smoke"

    def format(self) -> str:
        """:return: the canonical name; with ``k_legacy`` the five-field form is re-emitted."""
        base = f"{self.exp}_{self.dataset}_{self.backbone}_{self.adapt}"
        if self.k_legacy is not None:
            base = f"{base}_k{self.k_legacy}"
        return f"{base}_s{self.seed}"

    def __str__(self) -> str:
        return self.format()

    @classmethod
    def parse(cls, text: str) -> RunId:
        """Parse a ``run_id``, in current (4 fields) or historical (5 fields with ``k``) form.

        :param text: name to parse.
        :return: the identity; ``k_legacy`` is set only for the historical form.
        :raises ValueError: if the name has an unexpected shape, the fields before the seed
            are not four, or a value is outside the grammar.
        """
        m = _RUN_ID_RE.match(text)
        if not m:
            raise ValueError(
                f"malformed run_id: '{text}' "
                f"(expected {{exp}}_{{dataset}}_{{backbone}}_{{adapt}}_s{{S}}, "
                f"or the historical form with _k{{K}} before the seed)"
            )
        parts = m.group("rest").split("_")
        if len(parts) != 4:
            raise ValueError(
                f"malformed run_id: '{text}': expected 4 fields before the seed, found "
                f"{len(parts)} ({parts}). The adapt field uses '-', not '_'."
            )
        exp, dataset, backbone, adapt = parts
        return cls(
            exp=exp,
            dataset=dataset,
            backbone=backbone,
            adapt=adapt,
            seed=int(m.group("seed")),
            k_legacy=m.group("k"),
        )
