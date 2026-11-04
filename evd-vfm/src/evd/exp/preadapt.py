"""Pre-adaptation: a PEFT family arrives already trained from another run.

E3-C trains reparam and prompt jointly and merges only for inference. Two
sequential regimes are not contained in the joint one:

``fondi``
    LoRA/DoRA is already trained; it is merged into the backbone weights before the
    run starts and VPT is trained on top. The pre-trained family becomes the base
    model (E3-S, reparam -> prompt).
``congela``
    VPT is already trained; its prompts are loaded into a composition slot and frozen,
    and only the reparam part is trained; merging stays after training (E3-S,
    prompt -> reparam).

``fondi`` acts on the backbone before any ``apply()``; ``congela`` acts on a slot of an
already applied strategy (prompts must be applied last, see ``composite.py`` point 1).
Freezing keeps the two directions symmetric: in ``fondi`` the pre-trained family cannot
keep training, since it lives inside ``base.weight``. Pre-adaptation enters the
``config_sha``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from evd.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from evd.models.adapt.base import AdaptationStrategy
    from evd.models.backbones.base import VideoBackbone

log = get_logger(__name__)

__all__ = [
    "MODES",
    "PreadaptationSpec",
    "freeze_slot",
    "merge_into_backbone",
    "read_preadaptation",
    "prepare_backbone",
    "resolve_checkpoint",
]

#: The two modes, i.e. the two E3-S directions.
MODES: tuple[str, ...] = ("fondi", "congela")


@dataclass(frozen=True)
class PreadaptationSpec:
    """What arrives already trained, from where, and what to do with it.

    :param modo: ``"fondi"`` (merged into the weights before training) or ``"congela"``
        (installed and not trained).
    :param origine: checkpoint providing the weights, as written in the config: relative
        to the runs folder, never absolute (same rule as ``init_from``).
    :param adapt: ``configs/adapt/`` group that produced those weights; needed by
        ``fondi`` to rebuild the strategy before merging.
    :param slot: composition slot to freeze; needed by ``congela``.
    :param checkpoint_state: ``last`` for historical E3-S, ``selected`` for the best
        validation AUROC (E3-SV); never an implicit fallback.
    :raises ValueError: on an unknown mode, a missing origin, or a missing field required
        by the mode.
    """

    modo: str
    origine: str
    adapt: str | None = None
    slot: str | None = None
    checkpoint_state: str = "last"

    def __post_init__(self) -> None:
        if self.checkpoint_state not in {"last", "selected"}:
            raise ValueError("preadattamento.checkpoint_state must be last or selected")
        if self.modo not in MODES:
            raise ValueError(f"unknown preadattamento.modo {self.modo!r}: allowed {list(MODES)}")
        if not self.origine:
            raise ValueError("preadattamento without `from`: there is no run to inherit from")
        if self.modo == "fondi" and not self.adapt:
            raise ValueError(
                "preadattamento `fondi` without `adapt`: a checkpoint carries tensors, not the "
                "geometry explaining them, so without the group that produced them the "
                "adapter to rebuild and merge is unknown"
            )
        if self.modo == "congela" and not self.slot:
            raise ValueError(
                "preadattamento `congela` without `slot`: state WHICH component of the "
                "composition arrives already trained, or the wrong one would be frozen"
            )


def read_preadaptation(cfg: Mapping[str, Any]) -> PreadaptationSpec | None:
    """:param cfg: run config; reads the top-level ``preadattamento`` field.
    :return: the spec, or ``None`` if the run starts from a bare backbone.
    :raises ValueError: if the block carries fields nobody reads (see
        ``check_known_fields``).
    """
    group = cfg.get("preadattamento")
    if not group:
        return None
    fields = dict(group)
    unknown = sorted(set(fields) - {"modo", "from", "adapt", "slot", "checkpoint_state"})
    if unknown:
        raise ValueError(f"preadattamento: fields {unknown} are read by nobody")
    return PreadaptationSpec(
        modo=str(fields.get("modo", "")),
        origine=str(fields.get("from", "")),
        adapt=(str(fields["adapt"]) if fields.get("adapt") else None),
        slot=(str(fields["slot"]) if fields.get("slot") else None),
        checkpoint_state=str(fields.get("checkpoint_state", "last")),
    )


def resolve_checkpoint(percorso: str) -> Path:
    """Resolve a checkpoint declared relative to the runs folder.

    ``runs/`` lives in different places on different machines, and an absolute path in the
    ``config_sha`` would make the same cell differ across machines. Used by ``init_from``
    and by pre-adaptation.

    :param percorso: value declared in the config.
    :return: the existing file.
    :raises FileNotFoundError: naming the tried path.
    """
    from evd.utils import env

    file = Path(str(percorso))
    if not file.is_absolute() and not file.is_file():
        candidato = env.runs_dir() / str(percorso).removeprefix("runs/")
        if candidato.is_file():
            file = candidato
    if not file.is_file():
        raise FileNotFoundError(
            f"the declared checkpoint ({percorso}) does not exist in {file}: the run would start "
            f"from scratch while CLAIMING to inherit"
        )
    return file


def merge_into_backbone(
    spec: PreadaptationSpec,
    backbone: VideoBackbone,
    *,
    configs_dir: Path,
    context: Any,
    device: str,
) -> dict[str, Any]:
    """Merge an already trained family into the backbone weights, before the run.

    Builds the strategy from the declared group, applies it, restores the saved weights,
    merges and unloads it: afterwards ΔW lives in ``base.weight`` with no wrapper left. The
    merge is checked to have actually written the weights.

    :param spec: pre-adaptation in ``fondi`` mode.
    :param backbone: the backbone, modified in place.
    :param configs_dir: Hydra groups folder, to compose ``adapt``.
    :param context: the run ``AdaptContext``.
    :param device: where the parameters live.
    :return: provenance to write to ``meta.json``.
    :raises ValueError: if the declared group cannot be merged.
    :raises RuntimeError: if the merge did not change the backbone weights.
    """
    from evd.models.adapt.factory import build_adaptation
    from evd.train.checkpoint import restore_adapter_state

    file = resolve_checkpoint(spec.origine)
    weights, provenance = _load_adapter(file, checkpoint_state=spec.checkpoint_state)
    strategy = build_adaptation(_compose_adapt(Path(configs_dir), str(spec.adapt)), context)
    if not strategy.merge_parts():
        raise ValueError(
            f"preadattamento `fondi` on {spec.adapt!r}, which cannot be merged (`merge_parts()` "
            f"is empty): prompts lengthen the sequence and no weight can absorb them. "
            f"Use `congela` for such a family"
        )
    strategy.apply(backbone)
    restore_adapter_state(strategy, {k: v.to(device) for k, v in weights.items()})
    fingerprint = _fingerprint(backbone)
    strategy.merge_()
    removed = strategy.unload_()
    if _fingerprint(backbone) == fingerprint:
        raise RuntimeError(
            f"the pre-adaptation from {file} did not change a single byte of the backbone: the "
            f"cell would run as if that family were absent, under a misleading name"
        )
    log.info(
        "preadattamento FONDI from %s (run %s, step %s): %d tensors merged, %d wrappers removed",
        file,
        provenance["run_id"],
        provenance["step"],
        provenance["n_tensori"],
        removed,
    )
    return {"modo": "fondi", "adapt": spec.adapt, "wrapper_rimossi": int(removed), **provenance}


def prepare_backbone(
    cfg: Mapping[str, Any],
    backbone: VideoBackbone,
    *,
    context: Any,
    device: str,
) -> tuple[PreadaptationSpec | None, dict[str, Any] | None]:
    """Read the run pre-adaptation and, for ``fondi``, merge it into the backbone.

    The config-to-preadapted-backbone translation, shared by the runner and
    ``exp.demo.live_engine`` (demonstrator, external evaluation, explainability), so cells
    rebuilt outside the runner are not run on the bare backbone (``docs/09`` R24). Call it
    before any ``apply()`` of the cell strategy; ``congela`` does nothing here, since the
    frozen prompts are part of the saved adapter state.

    :param cfg: run config; reads ``preadattamento`` and, if present, ``configs_dir``
        (default ``configs``, relative to the working folder, then to the repository root).
    :param backbone: the backbone, modified in place in ``fondi`` mode.
    :param context: the run ``AdaptContext``.
    :param device: where the parameters live.
    :return: the spec (``None`` for a bare backbone) and the merge receipt for
        ``meta.json`` (``None`` if nothing was merged).
    :raises FileNotFoundError: if the source checkpoint or its ``adapt`` group is missing.
    :raises ValueError: if the block is malformed or the group cannot be merged.
    :raises RuntimeError: if the merge did not change the weights.
    """
    spec = read_preadaptation(cfg)
    if spec is None or spec.modo != "fondi":
        return spec, None
    configs_dir = Path(str(cfg.get("configs_dir", "configs")))
    if not configs_dir.is_absolute() and not configs_dir.is_dir():
        # The runner works from the repository root, where `configs` resolves; the demonstrator
        # and scripts may not.
        from evd.utils import env

        configs_dir = env.repo_root() / configs_dir
    receipt = merge_into_backbone(
        spec, backbone, configs_dir=configs_dir, context=context, device=device
    )
    return spec, receipt


def freeze_slot(
    spec: PreadaptationSpec, strategy: AdaptationStrategy, *, device: str
) -> dict[str, Any]:
    """Load the weights of a previous run into a slot and freeze it.

    :param spec: pre-adaptation in ``congela`` mode.
    :param strategy: the already applied composition.
    :param device: where the parameters live.
    :return: provenance, with ``params_congelati`` (to subtract from the declared trainable
        parameters).
    :raises TypeError: if the strategy is not a composition with slots.
    """
    from evd.train.checkpoint import restore_adapter_state

    component = getattr(strategy, "componente", None)
    if not callable(component):
        raise TypeError(
            f"preadattamento `congela` on {type(strategy).__name__}, which has no slots: the "
            f"already trained family and the one to train must coexist, so this cell "
            f"`adapt` must be a composition"
        )
    file = resolve_checkpoint(spec.origine)
    weights, provenance = _load_adapter(file, checkpoint_state=spec.checkpoint_state)
    below = component(str(spec.slot))
    # Restore on the component, not the composition: a VPT run has bare keys (`blocks.3...`)
    # while a composition prefixes them with the slot (`prompt/blocks.3...`).
    restore_adapter_state(below, {k: v.to(device) for k, v in weights.items()})
    frozen = 0
    for parameter in below.trainable_parameters():
        parameter.requires_grad_(False)
        frozen += int(parameter.numel())
    log.info(
        "preadattamento CONGELA slot %s from %s (run %s, step %s): %d tensors, %d parameters frozen",
        spec.slot,
        file,
        provenance["run_id"],
        provenance["step"],
        provenance["n_tensori"],
        frozen,
    )
    return {"modo": "congela", "slot": spec.slot, "params_congelati": frozen, **provenance}


def _load_adapter(
    file: Path, *, checkpoint_state: str = "last"
) -> tuple[dict[str, Any], dict[str, Any]]:
    """:return: the adapter weights and the identity of the run that produced them.

    E3-SV requires ``checkpoint_state="selected"`` (``selected_adapter`` and
    ``selected_step``, no fallback). The default ``last`` keeps the historical E3-S choice
    of reading ``adapter`` (last step), not the selected snapshot: on six of twelve sources
    the gap is 4 steps, on the others 924 to 3,684 steps. The bias has one direction (E3-S
    starts from a worse version of one family) and is declared in the thesis; numbers are
    in the campaign ``operations/eredita.json``.

    :raises ValueError: on an unknown state or a missing selected snapshot.
    """
    import torch

    data = torch.load(file, map_location="cpu", weights_only=True)
    if checkpoint_state not in {"last", "selected"}:
        raise ValueError(f"unknown checkpoint state: {checkpoint_state}")
    selected = checkpoint_state == "selected"
    key = "selected_adapter" if selected else "adapter"
    step = data.get("selected_step" if selected else "step")
    if selected and (not isinstance(step, int) or not 0 < step <= data["step"]):
        raise ValueError(f"{file}: missing or invalid selected_step: {step}")
    weights = dict(data.get(key) or {})
    if not weights:
        raise ValueError(
            f"{file} does not contain {key}: cannot inherit the {checkpoint_state} state; "
            "no fallback to another state is allowed"
        )
    identity = dict(data.get("identity") or {})
    return weights, {
        "checkpoint": str(file),
        "run_id": str(identity.get("run_id", "?")),
        "step": step,
        "checkpoint_state": checkpoint_state,
        "last_step": data.get("step"),
        "selected_value": data.get("selected_value") if selected else None,
        "n_tensori": len(weights),
    }


def _compose_adapt(configs_dir: Path, nome: str) -> dict[str, Any]:
    """Compose an ``adapt`` group as a run composes it, not by reading the raw YAML.

    Ablation configs inherit from a base through ``defaults``. Hydra is a singleton and
    this function runs inside :meth:`ExperimentRunner.run`, which ``campagna.py`` calls
    inside its own ``initialize_config_dir``: the caller instance is set aside and restored
    intact afterwards.

    :param configs_dir: Hydra groups folder of the run.
    :param nome: ``adapt`` group to compose.
    :return: the resolved group.
    :raises FileNotFoundError: naming the searched file.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    folder = (Path(configs_dir) / "adapt").resolve()
    path = folder / f"{nome}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"missing adapt group: {path}")
    suspended = GlobalHydra.instance().hydra if GlobalHydra.instance().is_initialized() else None
    if suspended is not None:
        GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(folder), version_base="1.3"):
            composite = compose(config_name=nome)
    finally:
        if suspended is not None:
            # Request the instance again: on exit Hydra puts back a deep copy of the instance found
            # on entry, so an earlier reference is detached from the singleton.
            global_ = GlobalHydra.instance()
            global_.clear()
            global_.initialize(suspended)
    return dict(OmegaConf.to_container(composite, resolve=True))  # type: ignore[arg-type]


def _fingerprint(backbone: VideoBackbone) -> str:
    """:return: a short SHA-256 of the weights, to check that the merge wrote."""
    import hashlib

    h = hashlib.sha256()
    for name, tensor in sorted(backbone.module().state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().to("cpu").float().numpy().tobytes())
    return h.hexdigest()[:16]
