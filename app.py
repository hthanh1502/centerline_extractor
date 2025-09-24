from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
from centerline_extractor import extract_centerline_and_junctions

app = Flask(__name__, static_folder='static')

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/process-image', methods=['POST'])


def process_image():
    # upload ảnh
    file = request.files['file']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    cv2.imwrite("temp.jpg", img)

    segments, junctions = extract_centerline_and_junctions("temp.jpg", debug=False)

    # Vẽ lên ảnh
    for seg in segments:
        pts_np = np.array(seg, dtype=np.int32)
        cv2.polylines(img, [pts_np], isClosed=False, color=(0, 255, 0), thickness=2)

    for p in junctions:
        cv2.circle(img, tuple(p), 4, (0, 0, 255), -1)

    # Lưu ảnh kết quả
    result_path = os.path.join(app.static_folder, "result.jpg")
    cv2.imwrite(result_path, img)

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
        "segments": convert(segments),
        "junctions": convert(junctions),
        "image_url": "/static/result.jpg"
    })

if __name__ == '__main__':
    app.run(debug=True)
    # app.run(debug=True, port=8000)

