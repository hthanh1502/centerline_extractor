import cv2
import numpy as np
import random
from skimage.morphology import skeletonize

# Đọc ảnh gốc
image_path = 'img/2.png'
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError("Không tìm thấy ảnh gốc. Kiểm tra lại đường dẫn.")

# Lọc màu
color_min = np.array([192, 72, 0], dtype=np.uint8)
color_max = np.array([243, 242, 219], dtype=np.uint8)
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

# Tìm nút giao
junctions = []
for y in range(1, skeleton.shape[0] - 1):
    for x in range(1, skeleton.shape[1] - 1):
        if skeleton[y, x] == 255:
            roi = skeleton[y-1:y+2, x-1:x+2]
            count = cv2.countNonZero(roi) - 1
            if count >= 3:
                junctions.append((x, y))

# Thêm điểm gần viền
height, width = skeleton.shape
border_margin = 1
for y in range(height):
    for x in range(width):
        if skeleton[y, x] == 255:
            if x < border_margin or y < border_margin or x >= width - border_margin or y >= height - border_margin:
                junctions.append((x, y))

# Vẽ nút giao
skeleton_color = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
for (x, y) in junctions:
    cv2.circle(skeleton_color, (x, y), 3, (0, 0, 255), -1)
cv2.imwrite('skeleton_junctions_only.jpg', skeleton_color)

# Truy vết đoạn
visited = np.zeros_like(skeleton, dtype=bool)
segments = []

def get_neighbors(x, y):
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < skeleton.shape[1] and 0 <= ny < skeleton.shape[0]:
                if skeleton[ny, nx] == 255:
                    neighbors.append((nx, ny))
    return neighbors

def trace_segment(start, next_pixel):
    path = [start]
    stack = [(next_pixel, start)]
    while stack:
        (cx, cy), prev = stack.pop()
        if visited[cy, cx]:
            continue
        visited[cy, cx] = True
        path.append((cx, cy))
        if (cx, cy) in junctions or cx == 0 or cy == 0 or cx == skeleton.shape[1]-1 or cy == skeleton.shape[0]-1:
            break
        neighbors = [n for n in get_neighbors(cx, cy) if n != prev and not visited[n[1], n[0]]]
        for n in neighbors:
            stack.append((n, (cx, cy)))
    return path

segments_seen = set()
for (x, y) in junctions:
    for (nx, ny) in get_neighbors(x, y):
        if not visited[ny, nx]:
            segment = trace_segment((x, y), (nx, ny))
            if len(segment) > 2:
                endpoints = tuple(sorted([segment[0], segment[-1]]))
                if endpoints not in segments_seen:
                    segments_seen.add(endpoints)
                    segments.append(segment)

print(f"Số đoạn trắng tìm được: {len(segments)}")

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

colored_segments = np.zeros((*skeleton.shape, 3), dtype=np.uint8)
used_colors = []
for segment in segments:
    color = [random.randint(50, 255) for _ in range(3)]
    used_colors.append(color)
    for (x, y) in segment:
        colored_segments[y, x] = color
for (x, y) in junctions:
    cv2.circle(colored_segments, (x, y), 3, (0, 0, 255), -1)
cv2.imwrite('colored_segments.jpg', colored_segments)

dots_image = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
sampled_segments = []
for segment in segments:
    simplified = simplify_segment(segment, epsilon=2.0)
    sampled_segments.append(simplified)
for sampled in sampled_segments:
    for (x, y) in sampled:
        cv2.circle(dots_image, (x, y), 2, (0, 255, 255), -1)
for (x, y) in junctions:
    cv2.circle(dots_image, (x, y), 3, (0, 0, 255), -1)
cv2.imwrite('skeleton_dots_grouped.jpg', dots_image)

overlay_image = img.copy()
for sampled in sampled_segments:
    for (x, y) in sampled:
        cv2.circle(overlay_image, (x, y), 2, (0, 0, 0), -1)
for (x, y) in junctions:
    cv2.circle(overlay_image, (x, y), 3, (0, 0, 255), -1)
cv2.imwrite('anh_goc_co_diem_dac_trung.jpg', overlay_image)

connected_image = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
for sampled in sampled_segments:
    color = [random.randint(50, 255) for _ in range(3)]
    for i in range(len(sampled) - 1):
        pt1 = tuple(sampled[i])
        pt2 = tuple(sampled[i + 1])
        cv2.line(connected_image, pt1, pt2, color, thickness=2)
for (x, y) in junctions:
    cv2.circle(connected_image, (x, y), 6, (0, 0, 255), -1)
cv2.imwrite('segments_connected_only.jpg', connected_image)

final_overlay = img.copy()
for sampled in sampled_segments:
    color = [random.randint(50, 255) for _ in range(3)]
    for i in range(len(sampled) - 1):
        pt1 = tuple(sampled[i])
        pt2 = tuple(sampled[i + 1])
        cv2.line(final_overlay, pt1, pt2, color, thickness=2)
for (x, y) in junctions:
    cv2.circle(final_overlay, (x, y), 6, (0, 0, 255), -1)
cv2.imwrite('segments_connected_on_original.jpg', final_overlay)

# Hiển thị tất cả ảnh bằng OpenCV
cv2.imwrite("Skeleton_Junctions.jpg", skeleton_color)
cv2.imwrite("Colored_Segments.jpg", colored_segments)
cv2.imwrite("Overlay_Image.jpg", overlay_image)
cv2.imwrite("Connected_Skeleton.jpg", connected_image)
cv2.imwrite("Segments_on_Original.jpg", final_overlay)
