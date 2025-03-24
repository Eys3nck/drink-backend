from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import threading
import urllib.request
from torchvision import models

MODEL_PATH = "drink_classifier.pt"
MODEL_URL = "https://www.dropbox.com/scl/fi/qc9a30u3gmo87itwdsmrx/drink_classifier.pt?rlkey=lpzwdvd2lv93ipvm7vebdsz1h&dl=1"

if not os.path.exists(MODEL_PATH):
    print("🔽 Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅ Model downloaded.")


app = Flask(__name__)
CORS(app)  # Allow cross-origin for browser access

latest_status = {"level": "full"}

@app.route("/status")
def status():
    return jsonify(latest_status)

@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img_bytes = file.read()
    npimg = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Run detection/classification
    results = model_detect(frame)
    boxes = results[0].boxes
    drink_status = "unknown"

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls = int(box.cls[0])
        label = model_detect.names[cls]

        if "cup" in label or "bottle" in label or "glass" in label:
            glass_roi = frame[int(y1):int(y2), int(x1):int(x2)]
            img = Image.fromarray(cv2.cvtColor(glass_roi, cv2.COLOR_BGR2RGB))
            input_tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                output = model_classify(input_tensor)
                _, predicted = torch.max(output, 1)
                drink_status = class_labels[predicted.item()]
                latest_status["level"] = drink_status
                break

    return jsonify({"level": drink_status})

# YOLO and classifier setup
model_detect = YOLO("yolov8n.pt")
model_classify = models.resnet18()
model_classify.fc = torch.nn.Linear(model_classify.fc.in_features, 3)
model_classify.fc = torch.nn.Linear(model_classify.fc.in_features, 3)
model_classify.load_state_dict(torch.load("drink_classifier.pt", map_location="cpu"))
model_classify.eval()

class_labels = ["empty", "half", "full"]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Start server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
