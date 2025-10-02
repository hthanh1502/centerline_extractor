import cv2
import numpy as np
import random
from skimage.morphology import skeletonize

def extract_centerline_and_junctions(image_path, debug=False):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Không tìm thấy ảnh gốc.")

    # Lọc màu
    color_min = np.array([170, 72, 0], dtype=np.uint8) # BGR
    color_max = np.array([230, 242, 219], dtype=np.uint8)
    mask = cv2.inRange(img, color_min, color_max)
    filtered = cv2.bitwise_and(img, img, mask=mask)

    # Chuyển sang ảnh xám và nhị phân
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Loại bỏ vùng nhỏ
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean_binary = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 500:
            clean_binary[labels == i] = 255

    # Skeleton hóa
    binary_bool = clean_binary > 0
    skeleton = skeletonize(binary_bool).astype(np.uint8) * 255

    height, width = skeleton.shape

    # Tìm nút giao
    junctions = []
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if skeleton[y, x] == 255:
                roi = skeleton[y-1:y+2, x-1:x+2]
                count = cv2.countNonZero(roi) - 1
                if count >= 3:
                    junctions.append((x, y))

    # Thêm điểm gần viền
    border_margin = 5
    for y in range(height):
        for x in range(width):
            if skeleton[y, x] == 255:
                if x < border_margin or y < border_margin or x >= width - border_margin or y >= height - border_margin:
                    junctions.append((x, y))

    # Tính degree để tìm điểm cuối
    degree = np.zeros_like(skeleton, dtype=np.uint8)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if skeleton[y, x] == 255:
                roi = skeleton[y-1:y+2, x-1:x+2]
                degree[y, x] = cv2.countNonZero(roi) - 1

    endpoints = [(x, y) for y in range(height) for x in range(width)
                 if skeleton[y, x] == 255 and degree[y, x] == 1]

    # Truy vết đoạn
    def get_neighbors(x, y):
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    if skeleton[ny, nx] == 255: #màu trắng
                        neighbors.append((nx, ny))
        return neighbors

    visited = np.zeros_like(skeleton, dtype=bool)
    segments = []
    segments_seen = set()

    def trace_segment(start, next_pixel):
        path = [start]
        stack = [(next_pixel, start)]
        while stack:
            (cx, cy), prev = stack.pop()
            if visited[cy, cx]:
                continue
            visited[cy, cx] = True
            path.append((cx, cy))
            if (cx, cy) in junctions or (cx, cy) in endpoints or cx == 0 or cy == 0 or cx == width - 1 or cy == height - 1:
                break
            neighbors = [n for n in get_neighbors(cx, cy) if n != prev and not visited[n[1], n[0]]]
            for n in neighbors:
                stack.append((n, (cx, cy)))
        return path

    for (x, y) in junctions + endpoints:
        for (nx, ny) in get_neighbors(x, y):
            if not visited[ny, nx]:
                segment = trace_segment((x, y), (nx, ny))
                if len(segment) > 2:
                    endpoints_pair = tuple(sorted([segment[0], segment[-1]]))
                    if endpoints_pair not in segments_seen:
                        segments_seen.add(endpoints_pair)
                        segments.append(segment)

    # Gộp nút giao
    def group_junctions(junctions, radius=5):
        grouped = []
        used = [False] * len(junctions)
        junctions = list(junctions)
        for i, (x1, y1) in enumerate(junctions):
            if used[i]:
                continue
            group = [(x1, y1)]
            used[i] = True
            for j, (x2, y2) in enumerate(junctions):
                if not used[j] and np.hypot(x1 - x2, y1 - y2) < radius:
                    group.append((x2, y2))
                    used[j] = True
            gx = int(np.mean([pt[0] for pt in group]))
            gy = int(np.mean([pt[1] for pt in group]))
            grouped.append((gx, gy))
        return grouped

    def simplify_segment(segment, epsilon=2.0):
        contour = np.array(segment, dtype=np.int32).reshape((-1, 1, 2))
        simplified = cv2.approxPolyDP(contour, epsilon, False)
        return [tuple(pt[0]) for pt in simplified]

    junctions = group_junctions(junctions, radius=5)

    sampled_segments = []
    for segment in segments:
        simplified = simplify_segment(segment, epsilon=2.0)
        sampled_segments.append(simplified)

    junction_midpoints = [tuple(seg[len(seg)//2]) for seg in sampled_segments if len(seg) >= 2]
    sorted_data = sorted(zip(sampled_segments, junction_midpoints), key=lambda item: (item[1][1], item[1][0]))
    sampled_segments, junction_midpoints = zip(*sorted_data) if sorted_data else ([], [])
    sampled_segments = list(sampled_segments)
    junction_midpoints = list(junction_midpoints)

    return sampled_segments, junctions, junction_midpoints
