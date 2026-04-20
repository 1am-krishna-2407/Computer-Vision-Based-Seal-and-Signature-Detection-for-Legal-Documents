import shutil
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from infer import (
    DEFAULT_MODEL,
    IMAGE_EXTENSIONS,
    PDF_EXTENSIONS,
    ensure_clean_dir,
    ensure_dir,
    prepare_inputs,
)


SUPPORTED_UPLOAD_TYPES = sorted(suffix.lstrip(".") for suffix in (IMAGE_EXTENSIONS | PDF_EXTENSIONS))


st.set_page_config(
    page_title="Seal & Signature Detection",
    page_icon="SV",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)


def save_uploaded_files(uploaded_files, upload_dir: Path):
    saved_paths = []
    for file in uploaded_files:
        suffix = Path(file.name).suffix.lower()
        if suffix not in (IMAGE_EXTENSIONS | PDF_EXTENSIONS):
            continue
        target = upload_dir / file.name
        with open(target, "wb") as handle:
            handle.write(file.getbuffer())
        saved_paths.append(target)
    return saved_paths


def summarize_labels(label_dir: Path):
    sig_docs = 0
    seal_docs = 0
    any_docs = 0
    sig_count = 0
    seal_count = 0

    for label_file in label_dir.glob("*.txt"):
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

    return {
        "files_with_any_detection": any_docs,
        "files_with_signature": sig_docs,
        "files_with_seal": seal_docs,
        "signature_detections": sig_count,
        "seal_detections": seal_count,
    }


def render_result_image(result):
    rendered = result.plot()
    if isinstance(rendered, np.ndarray):
        return Image.fromarray(rendered[..., ::-1]) if rendered.shape[-1] == 3 else Image.fromarray(rendered)
    return None


def render_sidebar_header():
    st.markdown(
        """
        <div class="sidebar-shell">
            <div class="sidebar-kicker">Vision Workspace</div>
            <div class="sidebar-title">Detection Controls</div>
            <div class="sidebar-copy">
                Upload your files and run seal and signature detection.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-kicker">Upload a file and compare the original with the annotated result.</div>
            <h1>Seal & Signature Detection</h1>
            <p>
                The interface is intentionally simple: choose files from the sidebar, run detection,
                and review the source and annotated outputs side by side.
            </p>
            <div class="hero-tags">
                <span>Images</span>
                <span>PDF</span>
                <span>Annotated View</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_cards():
    col1, col2, col3 = st.columns(3, gap="large")
    cards = [
        (
            "Upload",
            "Select one or more images, PDFs, or TIFF files from the sidebar.",
        ),
        (
            "Detect",
            "The app prepares each page and runs the trained model in the background.",
        ),
        (
            "Review",
            "See the original page and the annotated output together on the main screen.",
        ),
    ]

    for col, (title, copy) in zip((col1, col2, col3), cards):
        with col:
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="card-title">{title}</div>
                    <div class="card-copy">{copy}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_empty_state():
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-title">Start by adding one or more documents</div>
            <div class="empty-copy">
                The sidebar only contains uploads and the run button. After detection, the page will
                show original files beside their annotated outputs.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_results(manifest, results, save_dir: Path):
    st.markdown("### Review Outputs")
    st.success(f"Inference complete. Results saved to `{save_dir}`")

    for (source_path, prepared_path), result in zip(manifest, results):
        annotated_image = render_result_image(result)
        if annotated_image is None:
            continue

        st.markdown(
            f"""
            <div class="page-header">
                <div class="page-title">{source_path.name}</div>
                <div class="page-copy">Original file and annotated result</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        original_col, annotated_col = st.columns(2, gap="large")
        with original_col:
            st.markdown("**Original**")
            st.image(str(prepared_path), use_container_width=True)
            st.caption(f"Source: {source_path.name}")
        with annotated_col:
            st.markdown("**Annotated**")
            st.image(annotated_image, use_container_width=True)


def apply_theme():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(176, 44, 44, 0.14), transparent 30%),
                radial-gradient(circle at top right, rgba(24, 95, 148, 0.14), transparent 28%),
                linear-gradient(180deg, #f5f0e7 0%, #ece4d4 100%);
            color: #1d2935;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #132231 0%, #1f3345 100%);
        }
        [data-testid="stSidebar"] * {
            color: #f5efe2;
        }
        .sidebar-shell {
            padding: 0.4rem 0 0.8rem;
        }
        .sidebar-kicker {
            font-size: 0.76rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: #c9d7e3;
            margin-bottom: 0.35rem;
        }
        .sidebar-title {
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 0.4rem;
        }
        .sidebar-copy {
            font-size: 0.95rem;
            line-height: 1.5;
            color: #dbe5ec;
        }
        .sidebar-divider {
            height: 1px;
            background: rgba(255, 255, 255, 0.14);
            margin: 1rem 0 1.1rem;
        }
        .hero-panel,
        .info-card,
        .empty-state,
        .page-header {
            border: 1px solid rgba(18, 33, 47, 0.09);
            box-shadow: 0 16px 40px rgba(34, 47, 62, 0.08);
        }
        .hero-panel {
            background: linear-gradient(135deg, rgba(255, 249, 240, 0.95), rgba(250, 242, 228, 0.9));
            border-radius: 24px;
            padding: 1.8rem 1.8rem 1.6rem;
            margin-bottom: 1rem;
        }
        .hero-kicker {
            font-size: 0.78rem;
            letter-spacing: 0.2em;
            text-transform: uppercase;
            color: #9b2f2f;
            margin-bottom: 0.6rem;
            font-weight: 700;
        }
        .hero-panel h1 {
            margin: 0 0 0.7rem;
            font-size: 2.2rem;
            line-height: 1.1;
            color: #162330;
        }
        .hero-panel p {
            margin: 0;
            font-size: 1rem;
            line-height: 1.65;
            color: #304252;
            max-width: 52rem;
        }
        .hero-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1rem;
        }
        .hero-tags span {
            background: rgba(156, 47, 47, 0.08);
            color: #7b2020;
            border: 1px solid rgba(156, 47, 47, 0.12);
            padding: 0.38rem 0.72rem;
            border-radius: 999px;
            font-size: 0.84rem;
            font-weight: 600;
        }
        .info-card {
            background: rgba(255, 252, 246, 0.92);
            border-radius: 20px;
            padding: 1.05rem 1.1rem;
            min-height: 138px;
            margin: 0.6rem 0 1rem;
        }
        .card-title {
            font-size: 0.98rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #9c2f2f;
            margin-bottom: 0.45rem;
        }
        .card-copy {
            font-size: 0.97rem;
            line-height: 1.58;
            color: #2b3d4d;
        }
        .empty-state {
            background: rgba(255, 251, 244, 0.86);
            border-radius: 22px;
            padding: 1.2rem 1.3rem;
            margin-top: 0.45rem;
        }
        .empty-title {
            font-size: 1.02rem;
            font-weight: 700;
            color: #162330;
            margin-bottom: 0.35rem;
        }
        .empty-copy {
            font-size: 0.98rem;
            line-height: 1.6;
            color: #34485b;
        }
        .page-header {
            background: rgba(255, 250, 242, 0.88);
            border-radius: 18px;
            padding: 0.95rem 1rem;
            margin: 1rem 0 0.8rem;
        }
        .page-title {
            font-size: 1rem;
            font-weight: 700;
            color: #162330;
        }
        .page-copy {
            font-size: 0.92rem;
            color: #566777;
            margin-top: 0.2rem;
            word-break: break-word;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    apply_theme()

    with st.sidebar:
        render_sidebar_header()
        uploads = st.file_uploader(
            "Upload files",
            type=SUPPORTED_UPLOAD_TYPES,
            accept_multiple_files=True,
            help="Supported formats include images, PDF, TIFF, BMP, and WEBP.",
        )
        run = st.button("Run Detection", type="primary", use_container_width=True, disabled=not uploads)

    render_hero()
    render_feature_cards()

    if not uploads:
        render_empty_state()

    if not run:
        return

    with st.spinner("Preparing files and running inference..."):
        session_dir = Path(tempfile.mkdtemp(prefix="seal_sig_ui_"))
        upload_dir = session_dir / "uploads"
        prepared_dir = session_dir / "prepared_images"
        docs_dir = session_dir / "documents"

        ensure_dir(upload_dir)
        ensure_clean_dir(prepared_dir)
        ensure_dir(docs_dir)

        saved_paths = save_uploaded_files(uploads, upload_dir)
        if not saved_paths:
            st.error("No supported files were uploaded.")
            st.stop()

        try:
            prepared_images, manifest = prepare_inputs(saved_paths, prepared_dir, docs_dir)
        except Exception as exc:
            st.error(f"Failed to prepare inputs: {exc}")
            st.stop()

        model = load_model(str(DEFAULT_MODEL))
        results = model.predict(
            source=str(prepared_dir),
            imgsz=640,
            conf=0.25,
            device="cpu",
            save=True,
            save_txt=True,
            save_conf=True,
            project="runs/detect",
            name="streamlit_ui",
            exist_ok=True,
            workers=0,
        )

        if not results:
            st.error("No results were returned by the model.")
            st.stop()

        save_dir = Path(results[0].save_dir)
        label_dir = save_dir / "labels"
        summarize_labels(label_dir)

    render_results(manifest, results, save_dir)

    with st.expander("Cleanup temporary files"):
        if st.button("Delete temporary session files"):
            shutil.rmtree(session_dir, ignore_errors=True)
            st.write("Temporary files removed.")


if __name__ == "__main__":
    main()
