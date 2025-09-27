from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
import uuid
from centerline_extractor import extract_centerline_and_junctions

app = Flask(__name__, static_folder='static')

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/process-image', methods=['POST'])
def process_image():
    file = request.files.get('file')
    if file is None:
        return jsonify({"error": "No file uploaded"}), 400

    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    temp_path = "temp.jpg"
    cv2.imwrite(temp_path, img)

    segments, junctions, junction_midpoints = extract_centerline_and_junctions(
        temp_path,
        debug=False
    )

    # Vẽ lên ảnh
    vis = img.copy()
    for i, seg in enumerate(segments):
        pts_np = np.array(seg, dtype=np.int32)
        np.random.seed(i)  # để màu ổn định giữa các lần chạy
        color = tuple(int(c) for c in np.random.randint(0, 255, 3))
        cv2.polylines(vis, [pts_np], isClosed=False, color=color, thickness=2)

    for i, seg in enumerate(segments):
        if len(seg) < 2:
            continue  

        # Tính điểm giữa đoạn
        mid_idx = len(seg) // 2
        mid_point = tuple(seg[mid_idx])
        # mid_point = (mid_point[0], mid_point[1] - 10)


        label = f"{i + 1}"
        cv2.putText(vis, label, mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)

    for p in junctions:
        cv2.circle(vis, tuple(p), 4, (0, 0, 255), -1)

    for p in junction_midpoints:
        cv2.circle(vis, tuple(p), 6, (0, 255, 255), -1)

    result_path = os.path.join(app.static_folder, "result.jpg")
    cv2.imwrite(result_path, vis)

    def convert(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [convert(i) for i in obj]
        else:
            return obj

    token = str(uuid.uuid4())

    return jsonify({
        "token": token,
        "segments": convert(segments),
        "junctions": convert(junctions),
        "junction_midpoints": convert(junction_midpoints),
        "image_url": "/static/result.jpg"
    })

if __name__ == '__main__':
    app.run(debug=True)
