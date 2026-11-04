"""Install release checkpoints and verify SHA-256 before replacing destination files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from urllib.request import Request, urlopen


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "weights/weights-manifest.json").read_text(encoding="utf-8"))
    records = manifest["weights"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", choices=[r["run_id"] for r in records])
    parser.add_argument("--all", action="store_true", help="Download every listed checkpoint")
    parser.add_argument("--repository", default="apiantonio/V-JEPA_Event-Detection-Localization_with_PEFT")
    args = parser.parse_args()
    if not args.all and not args.model:
        for record in records:
            print(f"{record['run_id']}: {record['bytes'] / 1e6:.1f} MB")
        return
    project = root / "evd-vfm" if (root / "evd-vfm/src").exists() else root
    for record in records:
        if not args.all and record["run_id"] not in args.model:
            continue
        target = project / "runs" / record["run_id"] / "checkpoint/checkpoint.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        pending = target.with_suffix(".pt.part")
        url = f"https://github.com/{args.repository}/releases/download/models-v1/{record['asset']}"
        request = Request(url, headers={"User-Agent": "EVD-weight-downloader"})
        digest = hashlib.sha256()
        count = 0
        with urlopen(request, timeout=120) as response, pending.open("wb") as output:
            for block in iter(lambda: response.read(8 << 20), b""):
                output.write(block)
                digest.update(block)
                count += len(block)
        if count != record["bytes"] or digest.hexdigest() != record["sha256"]:
            raise ValueError(f"Checksum mismatch: {pending}; the existing checkpoint was not replaced")
        os.replace(pending, target)
        metadata = root / "weights" / "run-configs" / record["run_id"]
        for source in metadata.glob("*"):
            destination = target.parent.parent / source.name
            if source.is_file() and not destination.exists():
                shutil.copyfile(source, destination)
        print(f"Verified and installed {target}")


if __name__ == "__main__":
    main()
