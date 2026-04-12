import argparse
import math
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_SIGNATURE = 0
CLASS_STAMP = 1

PROJECT_ROOT = Path(__file__).resolve().parent
TOBACCO_DIR = PROJECT_ROOT / "data" / "raw" / "tobacco" / "images"
CEDAR_DIR = PROJECT_ROOT / "data" / "raw" / "cedar" / "signatures" / "full_org"
ROBOFLOW_IMG_DIR = PROJECT_ROOT / "data" / "raw" / "roboflow" / "train" / "images"
ROBOFLOW_LBL_DIR = PROJECT_ROOT / "data" / "raw" / "roboflow" / "train" / "labels"

EXTRACTED_STAMP_DIR = PROJECT_ROOT / "data" / "interim" / "stamp_crops"
SYNTH_IMAGES_DIR = PROJECT_ROOT / "data" / "synthetic" / "images"
SYNTH_LABELS_DIR = PROJECT_ROOT / "data" / "synthetic" / "labels"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build synthetic legal documents with one signature and one seal/stamp."
    )
    parser.add_argument("--num-images", type=int, default=3000)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stamp-padding", type=float, default=0.08)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def ensure_clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def list_images(path: Path):
    return sorted([p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])


def load_bgr(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    return image


def yolo_to_xyxy(parts, width, height, padding=0.0):
    _, xc, yc, w, h = map(float, parts[:5])
    x1 = (xc - w / 2.0) * width
    y1 = (yc - h / 2.0) * height
    x2 = (xc + w / 2.0) * width
    y2 = (yc + h / 2.0) * height

    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding

    x1 = max(0, int(math.floor(x1 - pad_x)))
    y1 = max(0, int(math.floor(y1 - pad_y)))
    x2 = min(width, int(math.ceil(x2 + pad_x)))
    y2 = min(height, int(math.ceil(y2 + pad_y)))
    return x1, y1, x2, y2


def crop_to_foreground(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    alpha = image[:, :, 3]
    foreground = cv2.bitwise_and(gray, alpha)
    coords = cv2.findNonZero((foreground > 0).astype(np.uint8))
    if coords is None:
        return image
    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y + h, x:x + w]


def white_to_alpha(image_bgr, threshold=245):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    alpha = np.where(gray < threshold, 255, 0).astype(np.uint8)
    rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    return crop_to_foreground(rgba)


def prepare_tobacco_background(path: Path, out_size):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load background: {path}")

    # Tobacco/YACCLAB pages are stored with black background and white foreground.
    image = cv2.bitwise_not(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.resize(image, (out_size, out_size), interpolation=cv2.INTER_AREA)

    if random.random() < 0.4:
        sigma = random.uniform(1.5, 6.0)
        noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if random.random() < 0.25:
        kernel = random.choice([3, 5])
        image = cv2.GaussianBlur(image, (kernel, kernel), 0)

    return image


def prepare_signature_asset(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load signature: {path}")

    image = cv2.bitwise_not(image)
    _, mask = cv2.threshold(image, 25, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    if coords is None:
        raise ValueError(f"Signature has no foreground: {path}")
    x, y, w, h = cv2.boundingRect(coords)
    image = image[y:y + h, x:x + w]
    mask = mask[y:y + h, x:x + w]

    ink = np.full((h, w, 3), 255, dtype=np.uint8)
    ink_color = random.randint(5, 45)
    ink[mask > 0] = (ink_color, ink_color, ink_color)
    rgba = cv2.cvtColor(ink, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask
    return rgba


def tint_stamp_asset(stamp_rgba):
    tinted = stamp_rgba.copy()
    color = random.choice(
        [
            (30, 30, random.randint(140, 210)),
            (40, 40, random.randint(120, 190)),
            (random.randint(100, 150), random.randint(40, 80), random.randint(150, 210)),
        ]
    )
    alpha_mask = tinted[:, :, 3] > 0
    tinted[alpha_mask, 0] = color[0]
    tinted[alpha_mask, 1] = color[1]
    tinted[alpha_mask, 2] = color[2]
    return tinted


def rotate_rgba(image, angle):
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2.0) - center[0]
    matrix[1, 2] += (new_h / 2.0) - center[1]
    return cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255, 0),
    )


def resize_rgba(image, max_width, max_height, min_scale=0.4, max_scale=1.0):
    h, w = image.shape[:2]
    scale_cap = min(max_width / max(w, 1), max_height / max(h, 1), max_scale)
    scale_floor = min(min_scale, scale_cap)
    scale = random.uniform(max(scale_floor, 0.05), max(scale_cap, 0.06))
    new_w = max(4, int(w * scale))
    new_h = max(4, int(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def intersects(box_a, box_b, padding=0):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return not (
        ax2 + padding < bx1
        or bx2 + padding < ax1
        or ay2 + padding < by1
        or by2 + padding < ay1
    )


def choose_region_position(canvas_w, canvas_h, obj_w, obj_h, region):
    margin = max(6, int(canvas_w * 0.02))
    if region == "signature":
        x_min = int(canvas_w * 0.32)
        x_max = canvas_w - obj_w - margin
        y_min = int(canvas_h * 0.64)
        y_max = canvas_h - obj_h - margin
    else:
        x_min = int(canvas_w * 0.22)
        x_max = canvas_w - obj_w - margin
        y_min = int(canvas_h * 0.42)
        y_max = canvas_h - obj_h - margin

    x_min = max(margin, min(x_min, x_max))
    y_min = max(margin, min(y_min, y_max))
    x_max = max(x_min, x_max)
    y_max = max(y_min, y_max)

    x = random.randint(x_min, x_max)
    y = random.randint(y_min, y_max)
    return x, y


def alpha_blend(background, overlay_rgba, x, y, opacity_range):
    h, w = overlay_rgba.shape[:2]
    roi = background[y:y + h, x:x + w].astype(np.float32)
    overlay_rgb = overlay_rgba[:, :, :3].astype(np.float32)
    alpha = (overlay_rgba[:, :, 3].astype(np.float32) / 255.0)
    alpha *= random.uniform(*opacity_range)
    alpha = alpha[..., None]
    blended = overlay_rgb * alpha + roi * (1.0 - alpha)
    background[y:y + h, x:x + w] = np.clip(blended, 0, 255).astype(np.uint8)
    return background


def place_asset(background, asset_rgba, region, existing_boxes, opacity_range):
    canvas_h, canvas_w = background.shape[:2]
    candidate = asset_rgba.copy()
    for _ in range(4):
        obj_h, obj_w = candidate.shape[:2]
        for _ in range(40):
            x, y = choose_region_position(canvas_w, canvas_h, obj_w, obj_h, region)
            box = (x, y, x + obj_w, y + obj_h)
            if all(not intersects(box, other, padding=12) for other in existing_boxes):
                alpha_blend(background, candidate, x, y, opacity_range)
                return box
        new_w = max(4, int(candidate.shape[1] * 0.85))
        new_h = max(4, int(candidate.shape[0] * 0.85))
        candidate = cv2.resize(candidate, (new_w, new_h), interpolation=cv2.INTER_AREA)
    raise RuntimeError(f"Failed to place asset in region={region}")


def box_to_yolo(box, width, height):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    xc = x1 + (w / 2.0)
    yc = y1 + (h / 2.0)
    return xc / width, yc / height, w / width, h / height


def extract_stamp_crops(output_dir: Path, padding: float):
    ensure_clean_dir(output_dir)
    crop_count = 0

    image_files = {path.stem: path for path in list_images(ROBOFLOW_IMG_DIR)}
    label_files = sorted(ROBOFLOW_LBL_DIR.glob("*.txt"))

    for label_path in tqdm(label_files, desc="Extracting stamp crops"):
        image_path = image_files.get(label_path.stem)
        if image_path is None:
            continue

        image = load_bgr(image_path)
        height, width = image.shape[:2]

        with open(label_path, "r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]

        for index, line in enumerate(lines):
            parts = line.split()
            if int(float(parts[0])) != CLASS_STAMP:
                continue

            x1, y1, x2, y2 = yolo_to_xyxy(parts, width, height, padding=padding)
            if x2 - x1 < 6 or y2 - y1 < 6:
                continue

            crop = image[y1:y2, x1:x2]
            crop_rgba = white_to_alpha(crop)
            if crop_rgba.shape[0] < 6 or crop_rgba.shape[1] < 6:
                continue

            out_name = f"{label_path.stem}_stamp_{index:02d}.png"
            cv2.imwrite(str(output_dir / out_name), crop_rgba)
            crop_count += 1

    return crop_count


def synthesize_dataset(num_images, img_size, stamp_crops, signature_files):
    ensure_clean_dir(SYNTH_IMAGES_DIR)
    ensure_clean_dir(SYNTH_LABELS_DIR)

    background_files = list_images(TOBACCO_DIR)
    if not background_files:
        raise ValueError("No tobacco background images were found.")
    if not stamp_crops:
        raise ValueError("No stamp crops were extracted from Roboflow.")
    if not signature_files:
        raise ValueError("No signature images were found in CEDAR.")

    for index in tqdm(range(num_images), desc="Generating synthetic pages"):
        background = prepare_tobacco_background(random.choice(background_files), img_size)

        signature = prepare_signature_asset(random.choice(signature_files))
        signature = rotate_rgba(signature, random.uniform(-8, 8))
        signature = resize_rgba(
            signature,
            max_width=int(img_size * 0.36),
            max_height=int(img_size * 0.16),
            min_scale=0.55,
            max_scale=1.0,
        )

        stamp = cv2.imread(str(random.choice(stamp_crops)), cv2.IMREAD_UNCHANGED)
        stamp = tint_stamp_asset(stamp)
        stamp = rotate_rgba(stamp, random.uniform(-18, 18))
        stamp = resize_rgba(
            stamp,
            max_width=int(img_size * 0.24),
            max_height=int(img_size * 0.22),
            min_scale=0.55,
            max_scale=1.0,
        )

        boxes = []
        signature_box = place_asset(
            background,
            signature,
            region="signature",
            existing_boxes=boxes,
            opacity_range=(0.72, 0.95),
        )
        boxes.append(signature_box)

        stamp_box = place_asset(
            background,
            stamp,
            region="stamp",
            existing_boxes=boxes,
            opacity_range=(0.45, 0.78),
        )

        if random.random() < 0.35:
            overlay = np.full(background.shape, 255, dtype=np.uint8)
            alpha = random.uniform(0.02, 0.06)
            background = cv2.addWeighted(background, 1.0 - alpha, overlay, alpha, 0)

        image_name = f"synthetic_{index:05d}.jpg"
        label_name = f"synthetic_{index:05d}.txt"
        cv2.imwrite(str(SYNTH_IMAGES_DIR / image_name), background)

        signature_line = " ".join(
            [str(CLASS_SIGNATURE)]
            + [f"{value:.6f}" for value in box_to_yolo(signature_box, img_size, img_size)]
        )
        stamp_line = " ".join(
            [str(CLASS_STAMP)]
            + [f"{value:.6f}" for value in box_to_yolo(stamp_box, img_size, img_size)]
        )

        with open(SYNTH_LABELS_DIR / label_name, "w", encoding="utf-8") as handle:
            handle.write(signature_line + "\n" + stamp_line + "\n")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    for path in [EXTRACTED_STAMP_DIR, SYNTH_IMAGES_DIR, SYNTH_LABELS_DIR]:
        ensure_dir(path.parent)

    if args.clean:
        for path in [EXTRACTED_STAMP_DIR, SYNTH_IMAGES_DIR, SYNTH_LABELS_DIR]:
            if path.exists():
                shutil.rmtree(path)

    stamp_count = extract_stamp_crops(EXTRACTED_STAMP_DIR, padding=args.stamp_padding)
    signature_files = list_images(CEDAR_DIR)
    stamp_crops = list_images(EXTRACTED_STAMP_DIR)

    print(f"Extracted {stamp_count} stamp crops")
    print(f"Found {len(signature_files)} signature assets")
    print(f"Generating {args.num_images} synthetic pages")

    synthesize_dataset(args.num_images, args.img_size, stamp_crops, signature_files)

    print(f"Synthetic images: {len(list_images(SYNTH_IMAGES_DIR))}")
    print(f"Synthetic labels: {len(list(SYNTH_LABELS_DIR.glob('*.txt')))}")


if __name__ == "__main__":
    main()
