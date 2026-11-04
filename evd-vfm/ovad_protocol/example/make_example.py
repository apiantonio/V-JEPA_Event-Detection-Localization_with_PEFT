"""Creates synthetic data in the OVAD format, to look at the files and try evaluate_ovad.py. They are not
real data: videos of random duration, half with anomalies (triples start, end, class) and half without,
and a fake model that gives higher anomaly scores to the chunks that see an anomaly, with noise.
Usage:  python example/make_example.py   (writes to example/data/)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

C, H = 8.0, 4.0
HERE = Path(__file__).resolve().parent / "data"
NOISE = 0.25
CLASSES = ["Abuse", "Arrest", "Assault", "Burglary", "Fighting", "Robbery", "Shooting", "Vandalism"]


def random_labels(rng: random.Random, n_videos: int, prefix: str) -> dict:
    """Videos of random duration: half with an anomaly (sometimes in two intervals), half without."""
    out = {}
    for i in range(n_videos):
        duration = round(rng.uniform(20, 150), 2)
        intervals = []
        if i % 2 == 0:
            a = rng.uniform(0, duration - 6)
            b = min(duration, a + rng.uniform(3, 25))
            intervals.append([round(a, 2), round(b, 2)])
            if rng.random() < 0.3 and b + 20 < duration:
                a2 = rng.uniform(b + 5, duration - 3)
                intervals.append([round(a2, 2), round(min(duration, a2 + rng.uniform(2, 10)), 2)])
        cls = rng.choice(CLASSES)
        out[f"{prefix}{i:03d}"] = {"duration": duration, "anomalies": [[a, b, cls] for a, b in intervals]}
    return out


def fake_scores(rng: random.Random, labels: dict, noise: float) -> dict:
    """A fake model: higher anomaly scores on the chunks that see an anomaly, with noise."""
    scores = {}
    for key, lab in labels.items():
        s, j = [], 0
        while j * H < lab["duration"]:                  # grid "start": chunk j = [j*H, j*H + C]
            x0, x1 = j * H, min(lab["duration"], j * H + C)
            sees = any(min(x1, b) - max(x0, a) > 0 for a, b, _ in lab["anomalies"])
            s.append(round(min(1.0, max(0.0, (0.55 if sees else 0.30) + rng.gauss(0, noise))), 4))
            j += 1
        scores[key] = s
    return {"hyperparameters": {"chunk_size": C, "asking_rate": H, "grid": "start"}, "scores": scores}


def write(name: str, obj: dict) -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / name).write_text(json.dumps(obj, indent=1))


def main() -> None:
    rng = random.Random(0)
    for part, n in (("test", 40), ("val", 30)):
        lab = random_labels(rng, n, f"{part}_video_")
        write(f"{part}_labels.json", lab)
        write(f"{part}_scores.json", fake_scores(rng, lab, NOISE))
    print(f"example data written to {HERE}")


if __name__ == "__main__":
    main()
