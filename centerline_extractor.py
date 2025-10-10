import cv2
import numpy as np
import random
import os
from skimage.morphology import skeletonize

def extract_centerline_and_junctions(image_path, debug=False):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Không tìm thấy ảnh gốc.")

    # Lọc màu đường (BGR)
    color_min_tong = np.array([0, 0, 0], dtype=np.uint8)
    color_max_tong = np.array([232, 236, 255], dtype=np.uint8)

    color_min_duong = np.array([170, 0, 0], dtype=np.uint8)
    color_max_duong = np.array([255, 229, 255], dtype=np.uint8)

    color_min_chu = np.array([0, 0, 0], dtype=np.uint8)
    color_max_chu = np.array([255, 180, 255], dtype=np.uint8)

    # Tạo các mask
    mask_duong_tong = cv2.inRange(img, color_min_tong, color_max_tong)
    mask_duong_raw = cv2.inRange(img, color_min_duong, color_max_duong)
    mask_chu = cv2.inRange(img, color_min_chu, color_max_chu)

    mask_duong = cv2.bitwise_and(mask_duong_raw, mask_duong_raw, mask=cv2.bitwise_not(mask_chu))
    mask_road_full = cv2.bitwise_and(mask_duong_tong, cv2.bitwise_not(mask_chu))
    mask_road_clean = cv2.bitwise_or(mask_duong_tong, mask_road_full)

    # Morphology để làm mịn và loại bỏ nhiễu
    kernel_noise = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_road_clean = cv2.morphologyEx(mask_road_clean, cv2.MORPH_OPEN, kernel_noise)

    # Thêm morphological closing để nối các đoạn gần nhau 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    mask_road_clean = cv2.morphologyEx(mask_road_clean, cv2.MORPH_CLOSE, kernel)

    # === Distance Transform + Bridging để nối các đoạn gần nhau ===
    dist = cv2.distanceTransform(mask_road_clean, cv2.DIST_L2, 5)
    _, dist_thresh = cv2.threshold(dist, 10, 255, cv2.THRESH_BINARY)
    dist_thresh = dist_thresh.astype(np.uint8)

    kernel_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask_bridge = cv2.dilate(dist_thresh, kernel_bridge, iterations=1)

    mask_road_clean = cv2.bitwise_or(mask_road_clean, mask_bridge)

    # Lưu ảnh mask
    os.makedirs("output", exist_ok=True)
    cv2.imwrite("output/mask_duong_tong.png", mask_duong_tong)
    cv2.imwrite("output/mask_duong.png", mask_duong)
    cv2.imwrite("output/mask_chu.png", mask_chu)
    cv2.imwrite("output/mask_road_clean.png", mask_road_clean)

    # Loại bỏ vùng nhỏ
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_road_clean, connectivity=8)
    clean_binary = np.zeros_like(mask_road_clean)
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
                    if skeleton[ny, nx] == 255:
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
                start, end = segment[0], segment[-1]
                length = np.hypot(end[0] - start[0], end[1] - start[1])
                if length > 10:
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

    if debug:
        cv2.imshow("Mask Duong Raw", mask_duong_raw)
        cv2.imshow("Mask Chu", mask_chu)
        cv2.imshow("Mask Duong", mask_duong)
        cv2.imshow("Clean Binary", clean_binary)
        cv2.imshow("Skeleton", skeleton)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return sampled_segments, junctions, junction_midpoints