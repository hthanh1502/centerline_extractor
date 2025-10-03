from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
import uuid
from datetime import datetime
from centerline_extractor import extract_centerline_and_junctions

app = Flask(__name__, static_folder='static')

IMG_DIR = os.path.join(app.static_folder, 'output', 'img')
os.makedirs(IMG_DIR, exist_ok=True)

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

    vis = img.copy()
    for i, seg in enumerate(segments):
        pts_np = np.array(seg, dtype=np.int32)
        np.random.seed(i)
        color = tuple(int(c) for c in np.random.randint(0, 255, 3))
        cv2.polylines(vis, [pts_np], isClosed=False, color=color, thickness=2)

    for i, seg in enumerate(segments):
        if len(seg) < 2:
            continue
        mid_idx = len(seg) // 2
        mid_point = tuple(seg[mid_idx])
        label = f"{i + 1}"
        cv2.putText(vis, label, mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)

    token = str(uuid.uuid4())
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"{date_str}_{token}.jpg"
    result_path = os.path.join(IMG_DIR, filename)
    # cv2.imwrite(result_path, vis)

    def convert(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [convert(i) for i in obj]
        else:
            return obj

    return jsonify({
        "token": token,
        "segments": convert(segments),
        "image_url": f"/static/output/img/{filename}"
    })


# @app.route('/view-result', methods=['POST'])
# def view_result_by_token():
#     data = request.get_json()
#     token = data.get("token")

#     if not token:
#         return jsonify({"error": "Thiếu token trong yêu cầu"}), 400

#     if not os.path.exists(MAP_PATH):
#         return jsonify({"error": "Không có ánh xạ token"}), 404

#     with open(MAP_PATH, "r") as f:
#         token_map = json.load(f)

#     filename = token_map.get(token)
#     if not filename:
#         return jsonify({"error": "Không tìm thấy kết quả với token này"}), 404

#     json_path = os.path.join(JSON_DIR, f"{token}.json")
#     if not os.path.exists(json_path):
#         return jsonify({"error": "Không tìm thấy dữ liệu phân tích"}), 404

#     with open(json_path, "r") as f:
#         analysis_data = json.load(f)

#     return jsonify({
#         "token": token,
#         "image_url": f"/static/output/img/{filename}",
#         "segments": analysis_data.get("segments", [])
#     })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
