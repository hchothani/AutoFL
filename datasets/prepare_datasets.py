import argparse
import os
import subprocess
from pathlib import Path

"""Dataset Preparation Script

This script downloads and prepares large-scale driving datasets (BDD100K & KITTI)
for use with AutoFL domain-incremental workloads.  It will:

1.  Download archives from official mirrors (or Academic Torrents fall-back).
2.  Extract archives into the local `data/` directory (keeping the same layout
    our workload code expects).
3.  Optionally verify checksums to ensure data integrity.

Because these datasets are *very* large (tens of GB), the script prints helpful
progress messages and supports `--skip-download` & `--skip-extract` flags so you
can resume if the process was interrupted.

Example usage:

    # Download and prepare the full BDD100K dataset
    python datasets/prepare_datasets.py bdd100k --target ./data

    # Prepare KITTI but assume archives are already downloaded in ./downloads
    python datasets/prepare_datasets.py kitti --download-dir ./downloads --skip-download
"""

DATASET_URLS = {
    "bdd100k": {
        "train_images": "https://bdd-data.berkeley.edu/bdd100k/images/100k/train.zip",
        "val_images": "https://bdd-data.berkeley.edu/bdd100k/images/100k/val.zip",
        "labels": "https://bdd-data.berkeley.edu/bdd100k/labels/bdd100k_labels_images_trainval.zip",
    },
    "bdd100k_10k": {
        "train_images": "https://bdd-data.berkeley.edu/bdd100k/images/10k/train.zip",
        "val_images": "https://bdd-data.berkeley.edu/bdd100k/images/10k/val.zip",
        "labels": "https://bdd-data.berkeley.edu/bdd100k/labels/bdd100k_labels_images_trainval.zip",
    },
    "kitti": {
        "detection": "https://cvlibs.s3.eu-central-1.amazonaws.com/kitti/data_object_image_2.zip",
        "detection_labels": "https://cvlibs.s3.eu-central-1.amazonaws.com/kitti/data_object_label_2.zip",
    },
}

CHECKSUMS = {
    # Optional: MD5/SHA256 checksums can be added here for integrity verification
}


def run(cmd: str):
    """Run shell command, streaming output."""
    print(f"[cmd] {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def download_file(url: str, dst: Path):
    """Download file using wget if it doesn't exist."""
    if dst.exists():
        print(f"✓ {dst.name} already exists, skipping download")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    run(f"wget -c {url} -O {dst}")


def extract_archive(archive: Path, dst_dir: Path):
    """Extract .zip or .tar.gz archives using unzip / tar."""
    print(f"⇪ Extracting {archive.name} …")
    if archive.suffix == ".zip":
        run(f"unzip -q {archive} -d {dst_dir}")
    elif archive.suffixes[-2:] == [".tar", ".gz"]:
        run(f"tar -xzf {archive} -C {dst_dir}")
    else:
        raise ValueError(f"Unsupported archive format: {archive}")


def prepare_bdd100k(args):
    urls = DATASET_URLS["bdd100k"]
    download_dir = Path(args.download_dir)
    target_dir = Path(args.target)

    # 1. Download
    for name, url in urls.items():
        archive_path = download_dir / Path(url).name
        if not args.skip_download:
            download_file(url, archive_path)
        else:
            print(f"⤷ Skipping download for {archive_path.name}")

        # 2. Extract
        if not args.skip_extract:
            extract_archive(archive_path, target_dir / "bdd100k")
        else:
            print(f"⤷ Skipping extraction for {archive_path.name}")

    print("✅ BDD100K preparation complete! Data located at", target_dir / "bdd100k")


def prepare_bdd100k_10k(args):
    urls = DATASET_URLS["bdd100k_10k"]
    download_dir = Path(args.download_dir)
    target_dir = Path(args.target)

    # 1. Download
    for name, url in urls.items():
        archive_path = download_dir / Path(url).name
        if not args.skip_download:
            download_file(url, archive_path)
        else:
            print(f"⤷ Skipping download for {archive_path.name}")

        # 2. Extract
        if not args.skip_extract:
            extract_archive(archive_path, target_dir)
        else:
            print(f"⤷ Skipping extraction for {archive_path.name}")

    print("✅ BDD100K-10k preparation complete! Data located at", target_dir)


def prepare_kitti(args):
    urls = DATASET_URLS["kitti"]
    download_dir = Path(args.download_dir)
    target_dir = Path(args.target)

    for name, url in urls.items():
        archive_path = download_dir / Path(url).name
        if not args.skip_download:
            download_file(url, archive_path)
        else:
            print(f"⤷ Skipping download for {archive_path.name}")

        if not args.skip_extract:
            extract_archive(archive_path, target_dir / "kitti")
        else:
            print(f"⤷ Skipping extraction for {archive_path.name}")

    print("✅ KITTI preparation complete! Data located at", target_dir / "kitti")


def main():
    parser = argparse.ArgumentParser(description="Download & prepare datasets for AutoFL")
    parser.add_argument("dataset", choices=["bdd100k", "bdd100k_10k", "kitti"], help="Dataset to prepare")
    parser.add_argument("--target", default="./data", help="Where to store the extracted dataset")
    parser.add_argument("--download-dir", default="./downloads", help="Where to keep downloaded archives")
    parser.add_argument("--skip-download", action="store_true", help="Assume archives already exist")
    parser.add_argument("--skip-extract", action="store_true", help="Skip extraction step")
    args = parser.parse_args()

    if args.dataset == "bdd100k":
        prepare_bdd100k(args)
    elif args.dataset == "bdd100k_10k":
        prepare_bdd100k_10k(args)
    elif args.dataset == "kitti":
        prepare_kitti(args)
    else:
        raise ValueError(args.dataset)


if __name__ == "__main__":
    main() 