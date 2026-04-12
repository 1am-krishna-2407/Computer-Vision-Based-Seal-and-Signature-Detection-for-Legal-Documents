from ultralytics import YOLO
import torch


def main():
    if torch.cuda.is_available():
        device = 0
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("⚠️ GPU not available, using CPU")

    model = YOLO("yolov8n.pt")

    model.train(
        data="seal_dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=device,
        name="seal_signature_model"
    )


if __name__ == "__main__":
    main()
