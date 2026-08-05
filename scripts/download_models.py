from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STYLEGAN_REPO = PROJECT_ROOT / "models" / "stylegan2-ada-pytorch"
NETWORK_PATH = PROJECT_ROOT / "models" / "stylegan2-ffhq-256x256.pkl"
STYLEGAN_URL = "https://github.com/NVlabs/stylegan2-ada-pytorch.git"
NETWORK_URL = (
    "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/"
    "versions/1/files/stylegan2-ffhq-256x256.pkl"
)
MINIMUM_NETWORK_BYTES = 100 * 1024 * 1024


def clone_stylegan() -> None:
    if STYLEGAN_REPO.exists():
        print(f"StyleGAN2-ADA source already exists: {STYLEGAN_REPO}")
        return
    if shutil.which("git") is None:
        raise RuntimeError("git is required to download StyleGAN2-ADA")
    STYLEGAN_REPO.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            STYLEGAN_URL,
            str(STYLEGAN_REPO),
        ],
        check=True,
    )


def download_network() -> None:
    if NETWORK_PATH.exists():
        print(f"FFHQ network already exists: {NETWORK_PATH}")
        return
    NETWORK_PATH.parent.mkdir(parents=True, exist_ok=True)
    partial_path = NETWORK_PATH.with_suffix(".pkl.part")

    def report(block_count: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = min(block_count * block_size, total_size)
        percent = downloaded * 100 / total_size
        print(f"\rDownloading FFHQ network: {percent:5.1f}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(NETWORK_URL, partial_path, reporthook=report)
        if partial_path.stat().st_size < MINIMUM_NETWORK_BYTES:
            raise RuntimeError(
                "Downloaded FFHQ checkpoint is unexpectedly small; retry the download."
            )
        partial_path.replace(NETWORK_PATH)
        print(f"\nSaved: {NETWORK_PATH}")
    finally:
        if partial_path.exists() and not NETWORK_PATH.exists():
            partial_path.unlink()


def warm_open_clip_cache() -> None:
    try:
        import open_clip
    except ImportError as exc:
        raise RuntimeError(
            "open_clip_torch is not installed. Run pip install -r requirements.txt."
        ) from exc
    print("Downloading/caching OpenCLIP ViT-B-32 weights...")
    open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device="cpu",
    )
    print("OpenCLIP weights are ready.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the production generator and optional CLIP weights."
    )
    parser.add_argument(
        "--with-clip",
        action="store_true",
        help="Also warm the OpenCLIP model cache.",
    )
    parser.add_argument(
        "--skip-network",
        action="store_true",
        help="Clone source only; do not download the FFHQ pickle.",
    )
    args = parser.parse_args()
    clone_stylegan()
    if not args.skip_network:
        download_network()
    if args.with_clip:
        warm_open_clip_cache()
    print("Model preparation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
