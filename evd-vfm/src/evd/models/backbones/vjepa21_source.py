"""Adapter to the Meta V-JEPA 2.1 source.

Construction replicates ``vjepa2/src/hub/backbones.py::_make_vjepa2_1_model`` without
downloading weights or building the predictor. The executed architecture is
``vjepa2/app/vjepa_2_1/models/vision_transformer.py``; the clone is neither
copied nor modified.
"""

from __future__ import annotations

import contextlib
import json
import sys
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn

from evd.utils import env
from evd.utils.hashing import file_sha256
from evd.utils.io import write_json


@dataclass(frozen=True)
class ArchSpec:
    """A released V-JEPA 2.1 variant.

    The table replicates ``ARCH_NAME_MAP`` and the public wrappers of
    ``vjepa2/src/hub/backbones.py:21-24,288-339``, verified by running the four variants.
    """

    name: str
    backbone_id: str
    checkpoint_file: str
    factory: str
    ckpt_key: str
    embed_dim: int
    depth: int
    num_heads: int
    n_output_distillation: int
    hierarchical_layers: tuple[int, ...]

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads


ARCHS: Mapping[str, ArchSpec] = {
    "vit_base": ArchSpec(
        name="vit_base",
        backbone_id="vjepa21b",
        checkpoint_file="vjepa2_1_vitb_dist_vitG_384.pt",
        factory="vit_base",
        ckpt_key="ema_encoder",
        embed_dim=768,
        depth=12,
        num_heads=12,
        n_output_distillation=1,
        hierarchical_layers=(2, 5, 8, 11),
    ),
    "vit_large": ArchSpec(
        name="vit_large",
        backbone_id="vjepa21l",
        checkpoint_file="vjepa2_1_vitl_dist_vitG_384.pt",
        factory="vit_large",
        ckpt_key="ema_encoder",
        embed_dim=1024,
        depth=24,
        num_heads=16,
        n_output_distillation=1,
        hierarchical_layers=(5, 11, 17, 23),
    ),
    "vit_giant": ArchSpec(
        name="vit_giant",
        backbone_id="vjepa21giant",
        checkpoint_file="vjepa2_1_vitg_384.pt",
        factory="vit_giant_xformers",
        ckpt_key="target_encoder",
        embed_dim=1408,
        depth=40,
        num_heads=22,
        n_output_distillation=4,
        hierarchical_layers=(9, 19, 29, 39),
    ),
    "vit_gigantic": ArchSpec(
        name="vit_gigantic",
        backbone_id="vjepa21gigantic",
        checkpoint_file="vjepa2_1_vitG_384.pt",
        factory="vit_gigantic_xformers",
        ckpt_key="target_encoder",
        embed_dim=1664,
        depth=48,
        num_heads=26,
        n_output_distillation=4,
        hierarchical_layers=(11, 23, 37, 47),
    ),
}


def default_checkpoint(arch: str, root: Path | None = None) -> Path:
    """Resolve architecture and canonical checkpoint together.

    Keeping the pair in :data:`ARCHS` prevents building one variant and loading the
    weights of another.

    :param arch: key of :data:`ARCHS`.
    :param root: checkpoint folder; ``None`` uses ``EVD_CKPT_DIR``.
    :return: path of the checkpoint released by Meta for the variant.
    :raises ValueError: if ``arch`` is not a supported variant.
    """
    try:
        spec = ARCHS[arch]
    except KeyError:
        raise ValueError(f"unknown arch {arch!r}; allowed: {sorted(ARCHS)}") from None
    base = Path(root) if root is not None else env.ckpt_dir()
    return base / spec.checkpoint_file


class UpstreamNotFound(RuntimeError):
    """The clone pointed to by ``EVD_VJEPA_DIR`` is not usable."""


class ArchitectureMismatch(RuntimeError):
    """The factory built a variant different from the declared one."""


def _prepend_once(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def ensure_upstream_importable(repo: Path | None = None) -> Path:
    """Check the clone and add both the clone and its parent to the path.

    The parent serves qualified ``vjepa2.app.*`` imports; the clone serves the absolute
    ``src.*`` imports used internally by Meta.

    :raises UpstreamNotFound: if the clone is missing or incomplete.
    """
    try:
        root = Path(repo) if repo is not None else env.vjepa_dir()
    except (FileNotFoundError, KeyError) as exc:
        raise UpstreamNotFound(f"EVD_VJEPA_DIR not usable: {exc}") from exc
    root = root.resolve()
    required = (
        root / "app" / "vjepa_2_1" / "models" / "vision_transformer.py",
        root / "src" / "hub" / "backbones.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if root.name.lower() != "vjepa2" or missing:
        details = f"; missing files: {missing}" if missing else ""
        raise UpstreamNotFound(
            f"EVD_VJEPA_DIR={root} is not the facebookresearch/vjepa2 clone"
            f" (the folder must be named 'vjepa2'){details}"
        )
    _prepend_once(root.parent)
    _prepend_once(root)
    return root


@contextlib.contextmanager
def _meta_factory_context(device: torch.device) -> Iterator[None]:
    """Build empty parameters despite the upstream ``linspace().item()``.

    ``VisionTransformer.__init__`` builds the drop-path vector with ``torch.linspace`` and
    calls ``.item()`` on each element, which is undefined on the ``meta`` device: only that
    linspace stays on CPU.
    """
    if device.type != "meta":
        yield
        return
    original = torch.linspace

    def cpu_linspace(*args: Any, **kwargs: Any) -> torch.Tensor:
        kwargs.setdefault("device", "cpu")
        return original(*args, **kwargs)

    torch.linspace = cpu_linspace
    try:
        with torch.device(device):
            yield
    finally:
        torch.linspace = original


def _validate_architecture(encoder: nn.Module, spec: ArchSpec) -> None:
    actual = {
        "embed_dim": getattr(encoder, "embed_dim", None),
        "depth": len(getattr(encoder, "blocks", ())),
        "num_heads": getattr(encoder, "num_heads", None),
    }
    expected = {
        "embed_dim": spec.embed_dim,
        "depth": spec.depth,
        "num_heads": spec.num_heads,
    }
    if actual != expected:
        raise ArchitectureMismatch(
            f"arch {spec.name!r}, factory {spec.factory!r}: expected {expected}, got {actual}"
        )
    head_dim = spec.embed_dim // spec.num_heads
    if head_dim != 64:
        raise ArchitectureMismatch(
            f"arch {spec.name!r}: head_dim={head_dim}, expected 64. "
            "Probably a factory without the _xformers suffix: the state dict would "
            "have the same shapes and strict=True would not notice."
        )
    layers = tuple(getattr(encoder, "hierarchical_layers", ()))
    if layers != spec.hierarchical_layers:
        raise ArchitectureMismatch(
            f"arch {spec.name!r}: hierarchical_layers={layers}, expected {spec.hierarchical_layers}"
        )


def build_encoder(
    arch: str,
    *,
    img_size: int,
    num_frames: int,
    device: str | torch.device | None = None,
    repo: str | Path | None = None,
) -> nn.Module:
    """Build the Meta encoder and check what ``strict=True`` cannot see.

    :raises ValueError: on an unknown arch or an invalid geometry.
    """
    ensure_upstream_importable(Path(repo) if repo is not None else None)
    try:
        spec = ARCHS[arch]
    except KeyError:
        raise ValueError(f"unknown arch {arch!r}; allowed: {sorted(ARCHS)}") from None
    if img_size <= 0 or num_frames <= 0:
        raise ValueError(f"img_size and num_frames must be positive: {img_size}, {num_frames}")
    if img_size % 16 or num_frames % 2:
        raise ValueError(
            f"geometry not divisible by the 2x16x16 tokenizer: size={img_size}, frames={num_frames}"
        )

    from vjepa2.app.vjepa_2_1.models import vision_transformer

    kwargs = {
        "patch_size": 16,
        "img_size": (img_size, img_size),
        "num_frames": num_frames,
        "tubelet_size": 2,
        "use_sdpa": True,
        "use_SiLU": False,
        "wide_SiLU": True,
        "uniform_power": False,
        "use_rope": True,
        "img_temporal_dim_size": 1,
        "interpolate_rope": True,
        "n_output_distillation": spec.n_output_distillation,
    }
    target = torch.device(device or "cpu")
    with _meta_factory_context(target):
        encoder = cast(nn.Module, getattr(vision_transformer, spec.factory)(**kwargs))
    if target.type != "meta":
        encoder = encoder.to(target)
    _validate_architecture(encoder, spec)
    encoder.eval()
    return encoder


def _clean_state_dict(state: Mapping[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for original, value in state.items():
        key = original.replace("module.", "").replace("backbone.", "")
        if key in cleaned:
            raise ValueError(f"checkpoint keys collide after cleaning: {original!r} -> {key!r}")
        cleaned[key] = value
    return cleaned


def load_encoder_state(
    encoder: nn.Module,
    ckpt: Path,
    *,
    key: str,
    mmap: bool = True,
) -> dict[str, Any]:
    """Load the encoder key with ``strict=True`` and return the diagnostics.

    :raises FileNotFoundError: if the checkpoint does not exist.
    :raises KeyError: if ``key`` is missing.
    :raises TypeError: if the checkpoint or the key is not a state dict.
    """
    path = Path(ckpt)
    if not path.is_file():
        raise FileNotFoundError(f"V-JEPA 2.1 checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", mmap=mmap, weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint {path} is not a mapping: {type(payload).__name__}")
    if key not in payload:
        present = ", ".join(sorted(map(str, payload.keys()))) or "(none)"
        raise KeyError(f"key {key!r} missing in {path}; available keys: {present}")
    raw = payload[key]
    if not isinstance(raw, Mapping):
        raise TypeError(f"checkpoint[{key!r}] is not a state dict: {type(raw).__name__}")
    state = _clean_state_dict(raw)
    incompatible = encoder.load_state_dict(state, strict=True)
    return {
        "checkpoint": str(path),
        "state_dict_key": key,
        "n_tensors": len(state),
        "strict": True,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def build_loaded_encoder(
    arch: str,
    *,
    checkpoint: Path,
    key: str,
    img_size: int,
    num_frames: int,
    device: str | torch.device,
    repo: str | Path | None = None,
    mmap: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build without initializing 300M+ random weights, then load with ``strict``.

    The Meta factory applies ``trunc_normal_`` to every projection; building on the meta
    device keeps structure and architecture checks without allocating, and ``to_empty``
    materializes the tensors once on the final device.

    :raises ValueError: if ``device`` is ``meta``.
    """
    target = torch.device(device)
    if target.type == "meta":
        raise ValueError("build_loaded_encoder requires a real device, not 'meta'")
    encoder = build_encoder(
        arch,
        img_size=img_size,
        num_frames=num_frames,
        device="meta",
        repo=repo,
    )
    encoder.to_empty(device=target)
    report = load_encoder_state(encoder, checkpoint, key=key, mmap=mmap)
    encoder.eval()
    return encoder, report


def checkpoint_sha(ckpt: Path) -> str:
    """Return the SHA256 of a checkpoint, cached in a sidecar checked on size and mtime (ns).

    :raises FileNotFoundError: if the checkpoint does not exist.
    """
    path = Path(ckpt)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint to hash not found: {path}")
    stat = path.stat()
    sidecar = Path(f"{path}.sha256")
    if sidecar.is_file():
        try:
            cached = json.loads(sidecar.read_text(encoding="utf-8"))
            digest = str(cached["sha256"])
            if (
                cached.get("size") == stat.st_size
                and cached.get("mtime_ns") == stat.st_mtime_ns
                and len(digest) == 64
                and all(char in "0123456789abcdef" for char in digest)
            ):
                return digest
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
            pass
    digest = file_sha256(path)
    write_json(
        sidecar,
        {
            "sha256": digest,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "checkpoint": path.name,
        },
    )
    return digest


def architecture_report(arch: str, encoder: nn.Module) -> dict[str, Any]:
    """Return a serializable architecture report, used by E0a."""
    spec = ARCHS[arch]
    return {
        "declared": asdict(spec),
        "actual": {
            "embed_dim": int(encoder.embed_dim),
            "depth": len(encoder.blocks),
            "num_heads": int(encoder.num_heads),
            "head_dim": int(encoder.embed_dim) // int(encoder.num_heads),
            "hierarchical_layers": list(encoder.hierarchical_layers),
        },
    }
