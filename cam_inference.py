from ultralytics import YOLO
import cv2
import os
from datetime import datetime

# Load the model
model_path = "runs/segment/train4/weights/best.pt"
model = YOLO(model_path)

# Start the webcam
cap = cv2.VideoCapture(0)

# Create folder for saving
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_folder = os.path.join("webcam_results", timestamp)
os.makedirs(output_folder, exist_ok=True)

frame_count = 0
detection_saved = False

print("\n[INFO] 🔄 Webcam running... Press 'q' to quit manually.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ ERROR: Cannot read webcam frame.")
        break

    results = model.predict(source=frame, conf=0.25, imgsz=640, stream=True)

    for r in results:
        annotated = r.plot()
        boxes = r.boxes
        masks = r.masks

        if boxes is not None and len(boxes) > 0 and not detection_saved:
            print("✅ Detected object(s)!")

            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]
                print(f"📦 {class_name} | Confidence: {confidence:.2f}")

            save_path = os.path.join(output_folder, f"detection_{frame_count}.jpg")
            cv2.imwrite(save_path, annotated)
            print(f"💾 Saved detection to: {save_path}")

            detection_saved = True
            cap.release()
            cv2.destroyAllWindows()
            exit()

        cv2.imshow("Detection", annotated)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("🛑 Manual exit.")
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Webcam closed. Exiting...")
print(f"Results saved in: {output_folder}")