# train.py

from ultralytics import YOLO

def train_model():
    model = YOLO("yolov8s-seg.pt")  # or continue from your best.pt
    model.train(
        data="dataset/dataset.yaml",  # path to your dataset.yaml
        epochs=50,
        imgsz=640,
        project="runs",
        name="blood_train",
        task="segment"
    )

if __name__ == "__main__":
    train_model()
