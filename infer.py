import argparse
import shutil
from pathlib import Path

import pypdfium2 as pdfium
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | PDF_EXTENSIONS

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "runs" / "detect" / "seal_signature_model4" / "weights" / "best.pt"
WORK_ROOT = PROJECT_ROOT / "inference_workdir"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run seal/signature detection on images or PDFs."
    )
    parser.add_argument("source", help="File or directory containing images or PDFs.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="multiformat_inference")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def ensure_clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def collect_supported_files(source: Path):
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    if source.is_file():
        files = [source]
    else:
        files = [path for path in source.rglob("*") if path.is_file()]

    supported = [path for path in files if path.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not supported:
        raise ValueError(f"No supported files found under: {source}")
    return sorted(supported)


def copy_image(source: Path, output_dir: Path):
    destination = output_dir / source.name
    shutil.copy2(source, destination)
    return [destination]


def render_pdf(source: Path, output_dir: Path):
    pdf = pdfium.PdfDocument(str(source))
    pages = []
    scale = 200 / 72

    for index in range(len(pdf)):
        bitmap = pdf[index].render(scale=scale)
        image = bitmap.to_pil()
        page_path = output_dir / f"{source.stem}-{index + 1}.png"
        image.save(page_path)
        pages.append(page_path)

    if not pages:
        raise RuntimeError(f"No pages rendered from PDF: {source}")
    return pages


def prepare_inputs(files, prepared_dir: Path, docs_dir: Path):
    prepared_images = []
    manifest = []

    for source in files:
        suffix = source.suffix.lower()
        doc_subdir = docs_dir / source.stem
        ensure_dir(doc_subdir)

        if suffix in IMAGE_EXTENSIONS:
            pages = copy_image(source, doc_subdir)
        elif suffix in PDF_EXTENSIONS:
            pages = render_pdf(source, doc_subdir)
        else:
            continue

        for page_path in pages:
            prepared_name = f"{source.stem}__{page_path.name}"
            prepared_path = prepared_dir / prepared_name
            shutil.copy2(page_path, prepared_path)
            prepared_images.append(prepared_path)
            manifest.append((source, prepared_path))

    if not prepared_images:
        raise RuntimeError("No images were prepared for inference.")
    return prepared_images, manifest


def summarize_results(output_dir: Path):
    label_dir = output_dir / "labels"
    if not label_dir.exists():
        print("No label outputs were written.")
        return

    label_files = sorted(label_dir.glob("*.txt"))
    sig_docs = 0
    seal_docs = 0
    any_docs = 0
    sig_count = 0
    seal_count = 0

    for label_file in label_files:
        lines = [line.strip() for line in label_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            continue
        any_docs += 1
        has_sig = False
        has_seal = False
        for line in lines:
            cls = line.split()[0]
            if cls == "0":
                sig_count += 1
                has_sig = True
            elif cls == "1":
                seal_count += 1
                has_seal = True
        sig_docs += int(has_sig)
        seal_docs += int(has_seal)

    print(f"Predicted files with any detection: {any_docs}")
    print(f"Predicted files with signatures: {sig_docs}")
    print(f"Predicted files with seals: {seal_docs}")
    print(f"Total signature detections: {sig_count}")
    print(f"Total seal detections: {seal_count}")


def main():
    args = parse_args()
    source = Path(args.source).resolve()
    model_path = Path(args.model).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    files = collect_supported_files(source)

    prepared_dir = WORK_ROOT / "prepared_images"
    docs_dir = WORK_ROOT / "documents"
    if args.clean:
        ensure_clean_dir(WORK_ROOT)
    ensure_clean_dir(prepared_dir)
    ensure_dir(docs_dir)

    prepared_images, manifest = prepare_inputs(files, prepared_dir, docs_dir)

    print(f"Source files found: {len(files)}")
    print(f"Prepared page images: {len(prepared_images)}")

    manifest_path = WORK_ROOT / "manifest.txt"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        for original, prepared in manifest:
            handle.write(f"{original}\t{prepared}\n")

    model = YOLO(str(model_path))
    results = model.predict(
        source=str(prepared_dir),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=True,
        save_txt=True,
        save_conf=True,
        project=args.project,
        name=args.name,
        exist_ok=True,
        workers=args.workers,
    )

    if not results:
        print("No inference results were returned.")
        return

    save_dir = Path(results[0].save_dir)
    print(f"Results saved to: {save_dir}")
    print(f"Prepared image manifest: {manifest_path}")
    summarize_results(save_dir)


if __name__ == "__main__":
    main()
