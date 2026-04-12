import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_NAMES = ["signature", "seal"]

PROJECT_ROOT = Path(__file__).resolve().parent

ROBOFLOW_IMAGES = PROJECT_ROOT / "data" / "raw" / "roboflow" / "train" / "images"
ROBOFLOW_LABELS = PROJECT_ROOT / "data" / "raw" / "roboflow" / "train" / "labels"
SYNTH_IMAGES = PROJECT_ROOT / "data" / "synthetic" / "images"
SYNTH_LABELS = PROJECT_ROOT / "data" / "synthetic" / "labels"

YOLO_ROOT = PROJECT_ROOT / "seal_dataset"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a clean YOLO dataset from synthetic pages and real Roboflow pages."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def ensure_clean_root(path: Path):
    if path.exists():
        shutil.rmtree(path)
    (path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (path / "images" / "val").mkdir(parents=True, exist_ok=True)
    (path / "images" / "test").mkdir(parents=True, exist_ok=True)
    (path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (path / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (path / "labels" / "test").mkdir(parents=True, exist_ok=True)


def list_images(directory: Path):
    if not directory.exists():
        return []
    return sorted([path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS])


def gather_pairs(image_dir: Path, label_dir: Path, prefix: str):
    pairs = []
    for image_path in list_images(image_dir):
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            pairs.append((prefix, image_path, label_path))
    return pairs


def copy_pair(image_path: Path, label_path: Path, split_name: str, prefix: str):
    image_name = f"{prefix}_{image_path.name}"
    label_name = f"{prefix}_{label_path.name}"

    image_dst = YOLO_ROOT / "images" / split_name / image_name
    label_dst = YOLO_ROOT / "labels" / split_name / label_name

    shutil.copy2(image_path, image_dst)
    shutil.copy2(label_path, label_dst)


def write_data_yaml():
    yaml_path = YOLO_ROOT / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {YOLO_ROOT}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                f"nc: {len(CLASS_NAMES)}",
                f"names: {CLASS_NAMES}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    random.seed(args.seed)

    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("Ratios must be positive and leave room for a test split.")

    ensure_clean_root(YOLO_ROOT)

    synthetic_pairs = gather_pairs(SYNTH_IMAGES, SYNTH_LABELS, "synthetic")
    real_pairs = gather_pairs(ROBOFLOW_IMAGES, ROBOFLOW_LABELS, "real")

    if not synthetic_pairs:
        raise ValueError("No synthetic image/label pairs found. Run dataset_generation_script.py first.")
    if not real_pairs:
        raise ValueError("No real Roboflow image/label pairs were found.")

    all_pairs = synthetic_pairs + real_pairs
    random.shuffle(all_pairs)

    total = len(all_pairs)
    train_end = int(total * args.train_ratio)
    val_end = int(total * (args.train_ratio + args.val_ratio))

    splits = {
        "train": all_pairs[:train_end],
        "val": all_pairs[train_end:val_end],
        "test": all_pairs[val_end:],
    }

    for split_name, items in splits.items():
        for prefix, image_path, label_path in items:
            copy_pair(image_path, label_path, split_name, prefix)

    write_data_yaml()

    print(f"Total pairs: {total}")
    print(f"Synthetic pairs: {len(synthetic_pairs)}")
    print(f"Real pairs: {len(real_pairs)}")
    print(f"Train: {len(splits['train'])}")
    print(f"Val: {len(splits['val'])}")
    print(f"Test: {len(splits['test'])}")
    print(f"Dataset written to: {YOLO_ROOT}")


if __name__ == "__main__":
    main()
