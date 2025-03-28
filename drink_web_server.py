from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms, models
from ultralytics import YOLO
import urllib.request

# Model download settings
MODEL_PATH = "drink_classifier.pt"
MODEL_URL = "https://www.dropbox.com/scl/fi/ajmgapj6cuzlhn26mu2c8/drink_classifier.pt?rlkey=0dfjp079qxvft6u79ssgi4svb&st=fz9avv15&dl=1"

if not os.path.exists(MODEL_PATH):
    print("🔽 Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅ Model downloaded.")

# Flask setup
app = Flask(__name__)
CORS(app, origins=["https://eys3nck.github.io"])

# Global state
latest_status = {"level": "full"}
registered_cameras = []

# Routes
@app.route("/status")
def status():
    return jsonify(latest_status)

@app.route("/upload", methods=["POST"])
def upload():
    print("✅ /upload hit!")

    if "image" not in request.files:
        print("❌ No image in request")
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img_bytes = file.read()
    npimg = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if frame is None:
        print("❌ Failed to decode image")
        return jsonify({"error": "Failed to decode image"}), 400

    print("📷 Image received and decoded")

    results = model_detect(frame)
    boxes = results[0].boxes
    drink_status = "unknown"

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls = int(box.cls[0])
        label = model_detect.names[cls]
        print(f"🎯 Detected object: {label}")

        if "cup" in label or "bottle" in label or "glass" in label:
            glass_roi = frame[int(y1):int(y2), int(x1):int(x2)]
            img = Image.fromarray(cv2.cvtColor(glass_roi, cv2.COLOR_BGR2RGB))
            input_tensor = transform(img).unsqueeze(0)

            with torch.no_grad():
                output = model_classify(input_tensor)
                _, predicted = torch.max(output, 1)
                drink_status = class_labels[predicted.item()]
                latest_status["level"] = drink_status

            print(f"🥤 Drink classified as: {drink_status}")
            break

    print("✅ Returning:", drink_status)
    return jsonify({"level": drink_status}), 200


@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    if not data or "id" not in data or "ip" not in data:
        return jsonify({"error": "Missing id or ip"}), 400

    camera_id = data["id"]
    camera_ip = data["ip"]

    # Update or append
    for cam in registered_cameras:
        if cam["id"] == camera_id:
            cam["ip"] = camera_ip
            break
    else:
        registered_cameras.append({"id": camera_id, "ip": camera_ip})

    print(f"📷 Registered camera: {camera_id} @ {camera_ip}")
    return jsonify({"status": "registered", "id": camera_id, "ip": camera_ip})

@app.route("/cameras", methods=["GET"])
def list_cameras():
    return jsonify(registered_cameras)

# Load models
model_detect = YOLO("yolov8n.pt", task="detect")
model_detect.fuse = lambda *args, **kwargs: model_detect  # Disable fuse()

model_classify = models.resnet18()
model_classify.fc = torch.nn.Linear(model_classify.fc.in_features, 3)
model_classify.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model_classify.eval()

class_labels = ["empty", "half", "full"]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Run the server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
