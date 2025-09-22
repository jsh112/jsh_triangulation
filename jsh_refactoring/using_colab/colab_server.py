# ==========================
# colab_server.py  (Colab/GPU)
# ==========================
from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# YOLO 모델 로드 (GPU)
model = YOLO("best_6.pt")  # Colab에 업로드 필요

def decode_b64_image(b64_str):
    img_bytes = base64.b64decode(b64_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def extract_holds(frame):
    res = model(frame)[0]
    holds = []
    if res.masks is None:
        return holds
    masks = res.masks.data
    boxes = res.boxes
    names = model.names
    for i in range(len(boxes)):
        cx = int((boxes.xyxy[i][0].item() + boxes.xyxy[i][2].item()) / 2)
        cy = int((boxes.xyxy[i][1].item() + boxes.xyxy[i][3].item()) / 2)
        cls_id = int(boxes.cls[i].item())
        class_name = names[cls_id]
        color = (0, 255, 0)  # 임시 색상
        holds.append({"cx": cx, "cy": cy, "hold_index": i, "color": color, "class_name": class_name})
    return holds

@app.route("/process_frame", methods=["POST"])
def process_frame():
    data = request.get_json()
    b64_frame = data["frame"]
    frame = decode_b64_image(b64_frame)
    holds = extract_holds(frame)
    return jsonify({"holds": holds})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
