"""Memory, prototype and learned-localizer maps of the automatic-localization campaign.

Token representation: final post-norm tokens, standardized and projected on the train
normal PCA-128 (``z``). P28 pairs are ``[z_t, z_{t+1} - z_t]`` at the same position, with
their own train standardization and PCA-128. Map functions are torch, so the same code
serves the validation calibration and the test stage; training lives here too.
"""

from __future__ import annotations

import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from evd.exp.localization_auto import ChunkResult, MethodOutput, S0Video, read_json
from evd.explain.concepts import prototype_maps
from evd.explain.localization import MaskRule, map_mask, native_boxes
from evd.types import VideoMeta

SEED = 42
MASS70 = MaskRule("mass", 0.7)
RELATIVE = ("P08-projection", "P19-one_representative", "P27-centroids")
ABSOLUTE = (
    "P24-bank25000",
    "P25-contexts5",
    "P26-gaussian1",
    "P28-transitions",
    "P29-slots128",
    "P30-linear",
    "P32-top10",
)
METHODS = RELATIVE + ABSOLUTE
CONTEXTS = 5
SLOTS, TOP_SLOTS, BOTTLENECK = 128, 8, 64


class MemAE(nn.Module):
    """P29: 128 learned slots, top-8 cosine addressing, linear decoder."""

    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(SEED)
        self.memory = nn.Parameter(torch.randn(SLOTS, dim, generator=generator) * 0.1)
        self.decoder = nn.Linear(dim, dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        similarity = (
            nn.functional.normalize(z, dim=-1) @ nn.functional.normalize(self.memory, dim=-1).T
        )
        values, index = similarity.topk(TOP_SLOTS, dim=-1)
        weights = values.softmax(-1)
        return self.decoder((weights[..., None] * self.memory[index]).sum(-2))


class Student(nn.Module):
    """P30: linear encoder to 64, linear decoder back to 128."""

    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        torch.manual_seed(SEED)
        self.encoder, self.decoder = nn.Linear(dim, BOTTLENECK), nn.Linear(BOTTLENECK, dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(z))


def mil_chunk_logits(head: nn.Linear, chunks: torch.Tensor) -> torch.Tensor:
    """P32: mean of the top 10% token logits of every chunk (``C × N × D`` → ``C``)."""
    logits = head(chunks.float()).squeeze(-1)
    k = math.ceil(0.1 * logits.shape[-1])
    return logits.topk(k, dim=-1).values.mean(-1)


def mil_video_logit(head: nn.Linear, chunks: torch.Tensor) -> torch.Tensor:
    """P32: mean of the top 10% chunk logits of a video."""
    chunk = mil_chunk_logits(head, chunks)
    return chunk.topk(math.ceil(0.1 * chunk.numel())).values.mean()


def train_reconstruction(
    model: nn.Module, train: torch.Tensor, val: torch.Tensor
) -> tuple[dict, dict]:
    """Adam 1e-3, batch 4096, at most 10 epochs, checkpoint of minimum validation MSE.

    :return: best state dict and the receipt (losses, chosen epoch, seconds).
    """
    torch.manual_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, history, start = None, [], time.perf_counter()
    for epoch in range(10):
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(SEED + epoch))
        for begin in range(0, len(train), 4096):
            batch = train[order[begin : begin + 4096]].to(next(model.parameters()).device)
            loss = ((model(batch) - batch) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            device = next(model.parameters()).device
            val_loss = float(
                torch.cat(
                    [((model(v.to(device)) - v.to(device)) ** 2).mean(-1) for v in val.split(65536)]
                ).mean()
            )
        history.append(val_loss)
        if best is None or val_loss < best[0]:
            best = (
                val_loss,
                epoch,
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            )
    return best[2], {
        "val_mse": history,
        "chosen_epoch": best[1],
        "fit_seconds": time.perf_counter() - start,
    }


def train_mil(
    train: list[tuple[torch.Tensor, int]], val: list[tuple[torch.Tensor, int]], device
) -> tuple[dict, dict]:
    """P32: one video per step, BCE on the video logit, checkpoint of minimum validation BCE."""
    torch.manual_seed(SEED)
    head = nn.Linear(train[0][0].shape[-1], 1).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    best, history, start = None, [], time.perf_counter()
    for epoch in range(10):
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(SEED + epoch))
        for i in order.tolist():
            chunks, label = train[i]
            loss = criterion(
                mil_video_logit(head, chunks).reshape(1),
                torch.tensor([float(label)], device=device),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            val_loss = float(
                np.mean(
                    [
                        float(
                            criterion(
                                mil_video_logit(head, c).reshape(1),
                                torch.tensor([float(y)], device=device),
                            )
                        )
                        for c, y in val
                    ]
                )
            )
        history.append(val_loss)
        if best is None or val_loss < best[0]:
            best = (
                val_loss,
                epoch,
                {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
            )
    return best[2], {
        "val_bce": history,
        "chosen_epoch": best[1],
        "fit_seconds": time.perf_counter() - start,
    }


def min_distance(z: torch.Tensor, bank: torch.Tensor, block: int = 2048) -> torch.Tensor:
    """:return: Euclidean distance of every row of ``z`` to its nearest bank row."""
    return torch.cat([torch.cdist(part, bank).amin(-1) for part in z.split(block)])


class MemoryModels:
    """Every fitted artifact of the memory family, on one device."""

    def __init__(self, fits: Path, device) -> None:
        def tensor(array) -> torch.Tensor:
            return torch.as_tensor(np.asarray(array), dtype=torch.float32, device=device)

        self.device = device
        pca = np.load(fits / "pca.npz")
        self.mean, self.scale, self.components = (
            tensor(pca[k]) for k in ("mean", "scale", "components")
        )
        self.binary = dict(np.load(fits / "binary_references.npz"))
        self.p24_bank = tensor(np.load(fits / "p24_bank.npy"))
        gaussian = np.load(fits / "p26_gaussian.npz")
        self.p26_mean, self.p26_var = tensor(gaussian["mean"]), tensor(gaussian["var"])
        pair = np.load(fits / "p28_pca.npz")
        self.p28_mean, self.p28_scale, self.p28_components = (
            tensor(pair[k]) for k in ("mean", "scale", "components")
        )
        self.p28_bank = tensor(np.load(fits / "p28_bank.npy"))
        context = np.load(fits / "p25_contexts.npz")
        self.p25_means = tensor(context["means"])
        self.p25_tokens = torch.as_tensor(context["tokens"], device=device)
        dim = self.components.shape[0]
        self.p29 = MemAE(dim).to(device)
        self.p29.load_state_dict(
            torch.load(fits / "p29.pt", map_location=device, weights_only=True)
        )
        self.p30 = Student(dim).to(device)
        self.p30.load_state_dict(
            torch.load(fits / "p30.pt", map_location=device, weights_only=True)
        )
        self.p32 = nn.Linear(self.components.shape[0], 1).to(device)
        self.p32.load_state_dict(
            torch.load(fits / "p32.pt", map_location=device, weights_only=True)
        )
        for model in (self.p29, self.p30, self.p32):
            model.eval()

    def project(self, tokens: torch.Tensor) -> torch.Tensor:
        """:return: ``N × 128`` standardized PCA coordinates of ``N × 1024`` tokens."""
        return ((tokens.float() - self.mean) / self.scale) @ self.components.T

    @torch.no_grad()
    def absolute_maps(
        self, z: torch.Tensor, mean_token: torch.Tensor, grid
    ) -> dict[str, np.ndarray]:
        """:return: the seven maps thresholded with a validation-calibrated value."""
        t, h, w = grid
        maps = {"P24-bank25000": min_distance(z, self.p24_bank)}
        similarity = nn.functional.normalize(self.p25_means, dim=-1) @ nn.functional.normalize(
            mean_token.float(), dim=0
        )
        nearest = similarity.topk(CONTEXTS).indices
        maps["P25-contexts5"] = min_distance(
            z, self.p25_tokens[nearest].reshape(-1, z.shape[-1]).float()
        )
        maps["P26-gaussian1"] = (((z - self.p26_mean) ** 2) / self.p26_var).mean(-1).sqrt()
        volume = z.reshape(t, h * w, -1)
        pairs = torch.cat([volume[:-1], volume[1:] - volume[:-1]], -1).reshape(-1, 2 * z.shape[-1])
        distance = min_distance(
            ((pairs - self.p28_mean) / self.p28_scale) @ self.p28_components.T, self.p28_bank
        )
        distance = distance.reshape(t - 1, h * w)
        transition = torch.zeros(t, h * w, device=z.device)
        transition[:-1] += distance
        transition[1:] += distance
        counts = torch.full((t, 1), 2.0, device=z.device)
        counts[0] = counts[-1] = 1.0
        maps["P28-transitions"] = (transition / counts).reshape(-1)
        maps["P29-slots128"] = ((self.p29(z) - z) ** 2).sum(-1)
        maps["P30-linear"] = ((self.p30(z) - z) ** 2).sum(-1)
        maps["P32-top10"] = torch.sigmoid(self.p32(z).squeeze(-1))
        return {k: v.reshape(grid).cpu().numpy().astype(np.float64) for k, v in maps.items()}


class MemoryStage:
    """P08, P19, P24–P30, P32 on the gated chunks; absolute thresholds from A1."""

    name = "memory"
    methods = METHODS

    def __init__(self, engine, spec, gate_fit_seconds: float, fits: Path) -> None:
        from evd.data.dataset import WindowItem
        from evd.exp.explanation_advanced import extract_features, prepare_clip, window_from_record

        self.engine, self.spec = engine, spec
        self.scorer = engine.scorer
        self.cuda = next(self.scorer.head.parameters()).is_cuda
        self.models = MemoryModels(fits, "cuda" if self.cuda else "cpu")
        self.thresholds = read_json(fits / "memory_thresholds.json")
        fit_times = read_json(fits / "memory_fit_seconds.json")
        self.fits = {name: gate_fit_seconds + fit_times[name] for name in self.methods}
        reference = read_json(fits / "p19_reference.json")["row"]
        paths = read_json(fits / "train_paths.json")
        clip = prepare_clip(
            engine, WindowItem(window_from_record(reference), paths[reference["video_id"]])
        )
        self.p19_reference, _ = extract_features(engine, clip)
        self.extract, self.prepare = extract_features, prepare_clip

    def parameters(self) -> dict:
        """:return: frozen parameters, part of the stage identity."""
        return {
            "relative_rule": "mass70",
            "absolute_thresholds": {k: self.thresholds[k]["threshold"] for k in ABSOLUTE},
            "contexts": CONTEXTS,
            "p19_ig": "schedule of the IG stage",
            "precision": self.scorer.precision,
        }

    def fit_seconds(self) -> dict[str, float]:
        """:return: gate plus each method's own fit and calibration."""
        return dict(self.fits)

    def begin_video(self, meta: VideoMeta, items: list, s0: S0Video) -> None:
        """No per-video state."""

    def _sync(self) -> float:
        if self.cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    def chunk(self, meta: VideoMeta, item, index: int) -> ChunkResult:
        """Compute the ten memory-family experiments on one gated chunk."""
        from evd.exp.localization_ig import IG_BATCH, IG_STEPS, IG_TOLERANCE
        from evd.explain.attribution import integrated_gradients

        start = self._sync()
        clip = self.prepare(self.engine, item)
        decoded = self._sync()
        feats, logit = self.extract(self.engine, clip)
        encoded = self._sync()
        grid = tuple(feats.grid)
        tokens = feats.tokens[0].float()
        outputs, saved = {}, {}

        def emit(name: str, values: np.ndarray, seconds: float, rule: MaskRule) -> None:
            post = time.perf_counter()
            boxes = native_boxes(map_mask(values, rule), meta.width, meta.height)
            outputs[name] = MethodOutput(boxes, seconds + time.perf_counter() - post)
            saved[name] = np.asarray(values, dtype=np.float32)

        begin = self._sync()
        direction = torch.as_tensor(
            self.models.binary["direction"], device=tokens.device, dtype=torch.float32
        )
        mean = torch.as_tensor(
            self.models.binary["mean"], device=tokens.device, dtype=torch.float32
        )
        # Same projection as evd.explain.concepts.direction_maps, without its sensitivity term.
        emit(
            "P08-projection",
            ((tokens - mean) @ direction).reshape(grid).cpu().numpy(),
            self._sync() - begin,
            MASS70,
        )

        begin = self._sync()
        difference = prototype_maps(tokens.cpu().numpy(), self.models.binary)[
            "prototype_difference"
        ]
        emit("P27-centroids", difference.reshape(grid), self._sync() - begin, MASS70)

        begin = self._sync()

        def head_forward(x, feats=feats):
            with self.scorer.autocast():
                return self.scorer.head(replace(feats, tokens=x, pooled=None))

        values, receipt = integrated_gradients(
            head_forward,
            feats.tokens,
            self.p19_reference.tokens,
            steps=IG_STEPS,
            batch_size=IG_BATCH,
            **IG_TOLERANCE,
        )
        emit(
            "P19-one_representative",
            values.sum(-1).reshape(grid).cpu().numpy(),
            self._sync() - begin,
            MASS70,
        )

        begin = self._sync()
        z = self.models.project(tokens)
        absolute = self.models.absolute_maps(z, tokens.mean(0), grid)
        shared = self._sync() - begin
        for name in ABSOLUTE:
            emit(
                name,
                absolute[name],
                shared,
                MaskRule("absolute", self.thresholds[name]["threshold"]),
            )
        return ChunkResult(
            outputs,
            maps=saved,
            receipts={"p19_ig": receipt, "absolute_maps_shared_seconds": shared},
            logit=logit,
            stage_decode_seconds=decoded - start,
            stage_encode_seconds=encoded - decoded,
        )
