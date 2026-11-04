"""Spatial grid of the annotator, consistent with ``in_pixel`` and the squash resize.

16x16 tokens live on the preprocessed frame, not on the original video pixels.
Contract and commands in ``docs/20_annotatore_spaziale.md``.
"""

from functools import lru_cache
from pathlib import Path

import yaml

#: Protocol of planes and grid when there is no source.
PROTOCOL_PATH = Path(__file__).resolve().parents[2] / "configs/protocol/edp_v2.yaml"


@lru_cache(maxsize=1)
def load_protocol() -> dict:
    """Read ``edp_v2`` once; the file has no interpolations to resolve.

    ``apps/`` does not import omegaconf (ADR-015): ``yaml.safe_load`` gives the same tree as
    ``OmegaConf.to_container(..., resolve=True)`` on this file.

    :return: the protocol as a dictionary, not to be modified.
    :raises ValueError: if the file contains interpolations that cannot be resolved here.
    """
    text = PROTOCOL_PATH.read_text(encoding="utf-8")
    if "${" in text:
        raise ValueError("The protocol contains interpolations: resolve them in evd/.")
    return yaml.safe_load(text)


def token_box(xyxy: list[int], grid: dict) -> dict:
    """Include every token sharing at least one native pixel with the box.

    :param xyxy: inclusive native coordinates within the video, not modified.
    :param grid: grid with half-open native edges, produced by ``token_grid``.
    :return: half-open token coordinates and their inclusive native projection.
    """

    def axis(start: int, end: int, edges: list[int]) -> tuple[int, int]:
        # The final box edge is inclusive; cell edges are exclusive.
        cells = [
            i
            for i, (left, right) in enumerate(zip(edges, edges[1:], strict=False))
            if left < right and left <= end and right > start
        ]
        return cells[0], cells[-1] + 1

    x0, x1 = axis(xyxy[0], xyxy[2], grid["x_edges"])
    y0, y1 = axis(xyxy[1], xyxy[3], grid["y_edges"])
    return {
        "xyxy": [x0, y0, x1, y1],
        "coordinate_convention": "token_xyxy_half_open",
        "grid": [grid["rows"], grid["columns"]],
        "patch_px": grid["patch_px"],
        "input_size": grid["input_size"],
        "rule": "intersecting_native_pixels",
        "pixel_xyxy": [
            grid["x_edges"][x0],
            grid["y_edges"][y0],
            grid["x_edges"][x1] - 1,
            grid["y_edges"][y1] - 1,
        ],
    }


def token_grid(video: dict, metadata: dict | None = None) -> dict:
    """Project token edges onto the native resolution, without crops.

    Native sources without their own grid use the ``edp_v2`` review grid; this does not
    attribute a token grid to the model that produced the boxes.

    :return: integer half-open edges, rounded as in ``explain.boxes.in_pixel``.
    :raises ValueError: if the geometry is not squash or incompatible with 16x16 tokens.
    """
    patch = 16
    source_grid = bool(metadata and ("size" in metadata or "grid" in metadata))
    if source_grid:
        if "size" not in metadata or "grid" not in metadata:
            raise ValueError("The source grid requires both size and grid.")
        size, variant = metadata["size"], metadata["geometry"]
        rows, columns = metadata["grid"][1:]
    else:
        geometry = load_protocol()["geometry"]
        size, variant = geometry["size"], geometry["preprocess_variant"]
        rows = columns = size // patch
        if metadata and metadata.get("geometry", variant) != variant:
            raise ValueError("The grid requires squash resize and 16x16 spatial tokens.")
    if variant != "squash" or rows * patch != size or columns * patch != size:
        raise ValueError("The grid requires squash resize and 16x16 spatial tokens.")
    return {
        "rows": rows,
        "columns": columns,
        "patch_px": patch,
        "input_size": size,
        "basis": "source" if source_grid else "edp_v2",
        "x_edges": [round(i * video["width"] / columns) for i in range(columns + 1)],
        "y_edges": [round(i * video["height"] / rows) for i in range(rows + 1)],
    }
