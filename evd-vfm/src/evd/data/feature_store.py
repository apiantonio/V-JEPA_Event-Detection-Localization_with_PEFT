"""Frozen-only feature store.

One safetensors file per ``(video, layer)`` plus a SQLite index. Array jobs write
separate ``index-N.sqlite`` shards, which are merged without concurrent writers on
the same database.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from evd.types import BackboneOutput, PreprocessSpec, WindowSpec
from evd.utils.hashing import stable_hash
from evd.utils.io import append_jsonl, ensure_dir, read_jsonl, temp_sibling, write_json
from evd.utils.logging import get_logger

FORMAT_VERSION = 1

# Fields that decide whether two features are comparable; mixing shards that differ
# in any of them would silently corrupt every downstream number. ``code_version`` is
# not among them: it is provenance (recorded in ``provenienza.jsonl``), while weights,
# preprocessing, compute regime and storage dtype determine the values.
IDENTITY_FIELDS = (
    "format_version",
    "backbone_id",
    "ckpt_sha",
    "preprocess",
    "preprocess_hash",
    "storage_dtype",
    "precision",
)

_LOG = get_logger(__name__)
_SAFE_VIDEO = re.compile(r"[^A-Za-z0-9._-]+")
_SCHEMA = """
CREATE TABLE IF NOT EXISTS features (
    key TEXT PRIMARY KEY,
    file TEXT NOT NULL,
    tensor_name TEXT NOT NULL,
    n_tokens INTEGER NOT NULL,
    dim INTEGER NOT NULL,
    grid_t INTEGER NOT NULL,
    grid_h INTEGER NOT NULL,
    grid_w INTEGER NOT NULL,
    layer INTEGER NOT NULL,
    dtype TEXT NOT NULL
)
"""


def feature_key(
    backbone_id: str,
    ckpt_sha: str,
    preprocess_hash: str,
    layer: int,
    window_id: str,
) -> str:
    """Return a stable SHA256 of the five identity fields of a feature.

    :param backbone_id: backbone identifier.
    :param ckpt_sha: checkpoint fingerprint.
    :param preprocess_hash: preprocessing fingerprint.
    :param layer: layer index.
    :param window_id: window identifier.
    :return: the hex digest.
    """
    return stable_hash(
        {
            "backbone_id": backbone_id,
            "ckpt_sha": ckpt_sha,
            "preprocess_hash": preprocess_hash,
            "layer": int(layer),
            "window_id": window_id,
        }
    )


@runtime_checkable
class FeatureReader(Protocol):
    """Read-only view consumed by ``WindowDataset``."""

    def __contains__(self, key: str) -> bool: ...

    def get(self, key: str) -> BackboneOutput | None: ...


@runtime_checkable
class FeatureStore(FeatureReader, Protocol):
    def key(
        self,
        backbone_id: str,
        ckpt_sha: str,
        pre: PreprocessSpec,
        layer: int,
        window: WindowSpec,
    ) -> str: ...

    def put(self, key: str, out: BackboneOutput) -> None: ...


class StoreMetadataMismatch(RuntimeError):
    """The existing store was produced with a different spec."""


def sizeof_estimate(
    n_windows: int,
    n_layers: int,
    tokens: int,
    dim: int,
    bytes_per_value: int = 2,
) -> int:
    """Estimate the token bytes (index and metadata are negligible at this scale).

    :param n_windows: number of windows.
    :param n_layers: number of layers.
    :param tokens: tokens per window.
    :param dim: token dimension.
    :param bytes_per_value: bytes per stored value.
    :return: estimated bytes.
    :raises ValueError: on non-positive arguments.
    """
    values = (n_windows, n_layers, tokens, dim, bytes_per_value)
    if any(value < 0 for value in values[:2]) or any(value <= 0 for value in values[2:]):
        raise ValueError(f"invalid arguments for sizeof_estimate: {values}")
    return n_windows * n_layers * tokens * dim * bytes_per_value


def _first_difference(expected: object, actual: object, prefix: str = "") -> str | None:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        for key in sorted(set(expected) | set(actual), key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected or key not in actual:
                return path
            difference = _first_difference(expected[key], actual[key], path)
            if difference is not None:
                return difference
        return None
    return prefix if expected != actual else None


def _video_filename(video_id: str) -> str:
    stem = Path(video_id).stem
    safe = _SAFE_VIDEO.sub("-", stem).strip("-.")
    if not safe:
        raise ValueError(f"video_id does not yield a safe file name: {video_id!r}")
    return f"{safe}.safetensors"


class SafetensorsFeatureStore:
    """Indexed feature store with explicit storage dtype and verified metadata."""

    def __init__(
        self,
        base_dir: str | Path,
        *,
        backbone_id: str,
        ckpt_sha: str,
        preprocess: PreprocessSpec,
        storage_dtype: str,
        precision: str,
        code_version: str,
        shard: int | None = None,
        create: bool = True,
    ) -> None:
        if storage_dtype not in {"fp16", "fp32"}:
            raise ValueError(
                f"storage_dtype={storage_dtype!r}: allowed fp16|fp32, never bf16. "
                "The value must be decided explicitly, not by a default."
            )
        # `precision` is the compute regime used to produce the features (distinct from the
        # on-disk `storage_dtype`); it is part of the metadata so that shards extracted with
        # different regimes cannot share a store.
        if precision not in {"fp32", "bf16"}:
            raise ValueError(
                f"precision={precision!r}: allowed fp32|bf16 (bf16 means AMP/autocast, "
                "never an encoder cast)"
            )
        if shard is not None and shard < 0:
            raise ValueError(f"shard must be >= 0, not {shard}")
        self.backbone_id = backbone_id
        self.ckpt_sha = ckpt_sha
        self.preprocess = preprocess
        self.storage_dtype = storage_dtype
        self.precision = precision
        self.code_version = code_version
        self.shard = shard
        self.root = Path(base_dir) / backbone_id / preprocess.hash()[:12]
        self.meta_path = self.root / "store_meta.json"
        self.provenance_path = self.root / "provenienza.jsonl"
        self.index_path = self.root / (
            f"index-{shard}.sqlite" if shard is not None else "index.sqlite"
        )
        self._locations: dict[str, tuple[str, int]] = {}
        # Read handles reused by `get` (index connection and last safetensors file).
        # Writers use `_database`, which really closes the handle (needed on Windows).
        self._rows: dict[str, tuple[str, str, int, int, int, int]] | None = None
        self._handle: Any = None
        self._current_file: tuple[Path, int, int] | None = None
        self._meta = {
            "format_version": FORMAT_VERSION,
            "backbone_id": backbone_id,
            "ckpt_sha": ckpt_sha,
            "preprocess": json.loads(json.dumps(preprocess.__dict__)),
            "preprocess_hash": preprocess.hash(),
            "storage_dtype": storage_dtype,
            "precision": precision,
            "code_version": code_version,
        }
        if self.meta_path.is_file():
            current = json.loads(self.meta_path.read_text(encoding="utf-8"))
            difference = _first_difference(
                {field: self._meta[field] for field in IDENTITY_FIELDS},
                {field: current.get(field) for field in IDENTITY_FIELDS},
            )
            if difference is not None:
                raise StoreMetadataMismatch(
                    f"inconsistent store_meta at field {difference!r}: "
                    f"expected={self._meta!r}, found={current!r}"
                )
            self._meta["code_version"] = str(current.get("code_version", code_version))
            if create and self._meta["code_version"] != code_version:
                self._record_provenance(code_version)
        elif create:
            ensure_dir(self.root)
            write_json(self.meta_path, self._meta)
        else:
            raise FileNotFoundError(f"store_meta.json not found in {self.root}")
        if create:
            with self._database() as connection:
                connection.execute(_SCHEMA)

    @classmethod
    def open_existing(
        cls,
        store_dir: str | Path,
        *,
        shard: int | None = None,
        create_index: bool = False,
    ) -> SafetensorsFeatureStore:
        """Open an existing store folder using only its ``store_meta.json``.

        :param store_dir: store folder.
        :param shard: shard index to write, or the merged index.
        :param create_index: whether to create a missing index.
        :return: the opened store.
        :raises FileNotFoundError: if ``store_meta.json`` is missing.
        :raises StoreMetadataMismatch: if the metadata resolve to another folder.
        """
        root = Path(store_dir)
        meta_path = root / "store_meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"store_meta.json not found in {root}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        preprocess = dict(meta["preprocess"])
        preprocess["mean"] = tuple(preprocess["mean"])
        preprocess["std"] = tuple(preprocess["std"])
        store = cls(
            root.parent.parent,
            backbone_id=str(meta["backbone_id"]),
            ckpt_sha=str(meta["ckpt_sha"]),
            preprocess=PreprocessSpec(**preprocess),
            storage_dtype=str(meta["storage_dtype"]),
            precision=str(meta["precision"]),
            code_version=str(meta["code_version"]),
            shard=shard,
            create=create_index,
        )
        if store.root.resolve() != root.resolve():
            raise StoreMetadataMismatch(
                f"inconsistent store layout: metadata resolve {store.root}, requested {root}"
            )
        return store

    def _connect(self, path: Path | None = None) -> sqlite3.Connection:
        connection = sqlite3.connect(path or self.index_path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _database(self, path: Path | None = None) -> Iterator[sqlite3.Connection]:
        """Open a transaction that really closes the handle (required by replace on Windows)."""
        connection = self._connect(path)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @property
    def metadata(self) -> dict[str, object]:
        """Return a serializable copy of the metadata read at opening.

        It contains the identity fields and the ``code_version`` of the store creator; later
        versions are listed by :meth:`code_versions`.
        """
        return dict(self._meta)

    def _record_provenance(self, code_version: str) -> None:
        """Record in ``provenienza.jsonl`` that a different code version is writing.

        The file is append-only, so concurrent array tasks never rewrite it.

        :param code_version: code version of this process.
        """
        _LOG.warning(
            "store %s created with code_version=%s, now written with %s: "
            "identity unchanged, drift recorded in %s",
            self.root,
            self._meta.get("code_version"),
            code_version,
            self.provenance_path.name,
        )
        append_jsonl(
            self.provenance_path,
            {
                "code_version": code_version,
                "shard": self.shard,
                "pid": os.getpid(),
                "ts": time.time(),
            },
        )

    def code_versions(self) -> tuple[str, ...]:
        """Return every code version that wrote to this store, without repetitions.

        :return: creator first, then the others in order of appearance.
        """
        views = [str(self._meta.get("code_version", ""))]
        for row in read_jsonl(self.provenance_path):
            version = str(row.get("code_version", ""))
            if version and version not in views:
                views.append(version)
        return tuple(v for v in views if v)

    def key(
        self,
        backbone_id: str,
        ckpt_sha: str,
        pre: PreprocessSpec,
        layer: int,
        window: WindowSpec,
    ) -> str:
        if backbone_id != self.backbone_id:
            raise StoreMetadataMismatch(f"backbone_id={backbone_id!r}, store={self.backbone_id!r}")
        if ckpt_sha != self.ckpt_sha:
            raise StoreMetadataMismatch(f"ckpt_sha={ckpt_sha!r}, store={self.ckpt_sha!r}")
        if pre.hash() != self.preprocess.hash():
            difference = _first_difference(pre.__dict__, self.preprocess.__dict__) or "preprocess"
            raise StoreMetadataMismatch(f"PreprocessSpec differs at field {difference!r}")
        result = feature_key(backbone_id, ckpt_sha, pre.hash(), layer, window.window_id)
        self._locations[result] = (window.video_id, int(layer))
        return result

    def __contains__(self, key: str) -> bool:
        """Return whether the store has ``key`` without opening SQLite.

        :param key: content-addressed window key.
        :return: whether the key is in the store.
        """
        if not self.index_path.is_file():
            return False
        return key in self._index()

    def get(self, key: str) -> BackboneOutput | None:
        """Return the feature of ``key``, or ``None`` if the store does not have it.

        The index connection and the last opened file are reused: windows of one video are
        consecutive and share a ``.safetensors`` file, so reopening per window would
        dominate evaluation time on a parallel filesystem.

        :param key: content-addressed window key.
        :return: tokens with their grid, or ``None``.
        :raises FileNotFoundError: if the index points to a missing file.
        """
        if not self.index_path.is_file():
            return None
        entry = self._index().get(key)
        if entry is None:
            return None
        file_name, tensor_name, grid_t, grid_h, grid_w, layer = entry
        tensors = self._opened(self.root / file_name)
        tokens = tensors.get_tensor(tensor_name)
        pooled_name = f"{tensor_name}__pooled"
        pooled = tensors.get_tensor(pooled_name) if pooled_name in tensors.keys() else None
        return BackboneOutput(
            tokens=tokens,
            grid=(grid_t, grid_h, grid_w),
            pooled=pooled,
            layer=layer,
        )

    def _index(self) -> dict[str, tuple[str, str, int, int, int, int]]:
        """Return the in-memory index, read once and then closed.

        Keeping a live connection would be slow on a parallel filesystem and would prevent
        deleting the folder on Windows; writers use :meth:`_database` and ``put``
        invalidates this copy.

        :return: key to (file, tensor name, grid, layer).
        """
        if self._rows is None:
            with self._database() as connection:
                self._rows = {
                    r["key"]: (
                        r["file"],
                        r["tensor_name"],
                        r["grid_t"],
                        r["grid_h"],
                        r["grid_w"],
                        r["layer"],
                    )
                    for r in connection.execute(
                        "SELECT key, file, tensor_name, grid_t, grid_h, grid_w, layer FROM features"
                    )
                }
        return self._rows

    def _opened(self, file_path: Path):
        """Return the safetensors handle of ``file_path``, reusing the last opened one.

        The handle identity is (path, mtime, size): an append replaces the whole file, and
        a handle kept by name would keep reading the previous memory-mapped version.

        :param file_path: safetensors file.
        :return: the ``safe_open`` object.
        :raises FileNotFoundError: if the file does not exist.
        """
        try:
            state = file_path.stat()
        except OSError as exc:
            raise FileNotFoundError(f"feature index points to a missing file: {file_path}") from exc
        identity = (file_path, state.st_mtime_ns, state.st_size)
        if self._current_file == identity and self._handle is not None:
            return self._handle
        if self._handle is not None:
            self._handle.__exit__(None, None, None)
            self._handle = None
        context = safe_open(file_path, framework="pt", device="cpu")
        self._handle = context
        self._handle.__enter__()
        self._current_file = identity
        return self._handle

    def close_store(self) -> None:
        """Release the file handle and forget the in-memory index (idempotent)."""
        if self._handle is not None:
            self._handle.__exit__(None, None, None)
            self._handle = None
        self._current_file = None
        self._rows = None

    def put(
        self,
        key: str,
        out: BackboneOutput,
        *,
        video_id: str | None = None,
        layer: int | None = None,
    ) -> None:
        if key in self:
            return
        remembered = self._locations.get(key)
        if remembered is not None:
            video_id = video_id or remembered[0]
            layer = remembered[1] if layer is None else layer
        if video_id is None or layer is None:
            raise ValueError(
                "put requires video_id/layer or a key obtained from store.key(...); "
                "the digest alone does not carry the on-disk layout."
            )
        self.put_many({key: out}, video_id=video_id, layer=layer)

    def put_many(
        self,
        entries: Mapping[str, BackboneOutput],
        *,
        video_id: str,
        layer: int,
    ) -> int:
        """Write all windows of one video/layer with a single atomic replace.

        :param entries: window key to backbone output.
        :param video_id: video identifier.
        :param layer: layer index.
        :return: number of written windows.
        :raises ValueError: if an output does not have batch size 1.
        """
        pending = {key: out for key, out in entries.items() if key not in self}
        if not pending:
            return 0
        for key, out in pending.items():
            if out.layer != layer:
                raise ValueError(f"feature {key}: out.layer={out.layer}, destination L{layer}")
            if out.tokens.shape[0] != 1:
                raise ValueError(
                    f"per-window feature store requires batch=1, got {out.tokens.shape[0]}"
                )
        relative = Path(f"L{layer}") / _video_filename(video_id)
        target = self.root / relative
        ensure_dir(target.parent)
        tensors: dict[str, torch.Tensor] = {}
        if target.is_file():
            with safe_open(target, framework="pt", device="cpu") as current:
                tensors.update({name: current.get_tensor(name) for name in current.keys()})
        dtype = torch.float16 if self.storage_dtype == "fp16" else torch.float32
        for key, out in pending.items():
            tensors[key] = out.tokens.detach().to(device="cpu", dtype=dtype).contiguous()
            if out.pooled is not None:
                tensors[f"{key}__pooled"] = (
                    out.pooled.detach().to(device="cpu", dtype=dtype).contiguous()
                )
        self._save_atomic(target, tensors)

        rows = []
        for key, out in pending.items():
            rows.append(
                (
                    key,
                    relative.as_posix(),
                    key,
                    int(out.tokens.shape[1]),
                    int(out.tokens.shape[2]),
                    *map(int, out.grid),
                    int(layer),
                    self.storage_dtype,
                )
            )
        with self._database() as connection:
            connection.executemany(
                """
                INSERT INTO features
                    (key, file, tensor_name, n_tokens, dim, grid_t, grid_h, grid_w, layer, dtype)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        # Writing invalidates what readers hold: the in-memory index misses the new keys and
        # the file was replaced.
        self.close_store()
        return len(rows)

    @staticmethod
    def _save_atomic(path: Path, tensors: Mapping[str, torch.Tensor]) -> None:
        # Unique temporary name per writer.
        tmp = temp_sibling(path)
        try:
            save_file(dict(tensors), str(tmp))
            # On Windows ``os.fsync`` on a read-only descriptor can fail with EBADF; ``r+b`` gives
            # a syncable handle without modifying the payload.
            with open(tmp, "r+b") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def count(self) -> int:
        if not self.index_path.is_file():
            return 0
        with self._database() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM features").fetchone()[0])

    def layers(self) -> tuple[int, ...]:
        """Return the layers actually indexed, in order.

        :return: layer indices present (empty if there is no index).
        """
        if not self.index_path.is_file():
            return ()
        with self._database() as connection:
            rows = connection.execute("SELECT DISTINCT layer FROM features ORDER BY layer")
            return tuple(int(row[0]) for row in rows)

    def known_keys(self) -> set[str]:
        """Return the keys of the whole store, not only of this shard's index.

        Writers of an append must know what any shard already extracted, while
        ``__contains__`` and :meth:`keys` describe what ``get`` can read from this index.

        :return: union of the keys of every ``index*.sqlite`` in the root.
        """
        keys: set[str] = set()
        for path in sorted(self.root.glob("index*.sqlite")):
            with self._database(path) as connection:
                keys.update(
                    str(row["key"]) for row in connection.execute("SELECT key FROM features")
                )
        return keys

    def keys(self) -> list[str]:
        """Return the indexed keys in stable order (used to compare or verify whole stores)."""
        if not self.index_path.is_file():
            return []
        with self._database() as connection:
            rows = connection.execute("SELECT key FROM features ORDER BY key")
            return [str(row["key"]) for row in rows]

    def verify_index(self) -> dict[str, float | int]:
        """Check that every index row points to an existing tensor, grouping by file.

        A missing index yields ``hit_rate = 0.0``: it is a store that cannot be read yet.

        :return: verification counters and hit rate.
        """
        if not self.index_path.is_file():
            return {"n_indexed": 0, "n_available": 0, "hit_rate": 0.0}
        with self._database() as connection:
            rows = list(connection.execute("SELECT file, tensor_name FROM features"))
        expected_by_file: dict[str, set[str]] = {}
        for row in rows:
            expected_by_file.setdefault(str(row["file"]), set()).add(str(row["tensor_name"]))
        available = 0
        for relative, expected in expected_by_file.items():
            path = self.root / relative
            if not path.is_file():
                continue
            with safe_open(path, framework="pt", device="cpu") as tensors:
                available += len(expected.intersection(tensors.keys()))
        total = len(rows)
        return {
            "n_indexed": total,
            "n_available": available,
            "hit_rate": available / total if total else 1.0,
        }

    def merge_shards(self, num_shards: int) -> int:
        """Merge the shard indices; duplicate keys reveal non-disjoint sharding.

        :param num_shards: expected number of shards.
        :return: number of merged rows.
        :raises FileNotFoundError: if a shard index is missing.
        :raises ValueError: on duplicates or invalid arguments.
        """
        if self.shard is not None:
            raise ValueError("merge_shards must be called on a store without shard")
        if num_shards <= 0:
            raise ValueError(f"num_shards must be > 0, not {num_shards}")
        shard_paths = [self.root / f"index-{index}.sqlite" for index in range(num_shards)]
        missing = [str(path) for path in shard_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing shard indices: {missing}")
        tmp = self.root / "index.sqlite.tmp"
        tmp.unlink(missing_ok=True)
        try:
            with self._database(tmp) as merged:
                merged.execute(_SCHEMA)
                for shard_path in shard_paths:
                    with self._database(shard_path) as shard_db:
                        rows: Iterable[sqlite3.Row] = shard_db.execute(
                            "SELECT key, file, tensor_name, n_tokens, dim, grid_t, grid_h, "
                            "grid_w, layer, dtype FROM features"
                        )
                        try:
                            merged.executemany(
                                "INSERT INTO features VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                (tuple(row) for row in rows),
                            )
                        except sqlite3.IntegrityError as exc:
                            raise ValueError(
                                f"duplicate key while merging {shard_path.name}: "
                                "the shards are not disjoint"
                            ) from exc
            os.replace(tmp, self.index_path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return self.count()
