import cv2
import numpy as np
import random
from skimage.morphology import skeletonize

# --- Cấu hình đường dẫn ảnh và tham số ---
image_path = 'img/h1.png'   # thay bằng đường dẫn thật nếu cần
color_min = (192, 72, 0)    # BGR min cho mask màu
color_max = (243, 242, 219) # BGR max cho mask màu
min_area = 500              # loại bỏ vùng nhỏ
sample_spacing = 5          # khoảng cách lấy mẫu điểm trên mỗi đoạn (pixel)
group_radius = 5            # bán kính gộp các junction gần nhau

# --- Đọc ảnh ---
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Không tìm thấy ảnh tại: {image_path}")

# --- Tạo mask theo khoảng màu và áp dụng ---
mask_color = cv2.inRange(img, np.array(color_min, dtype=np.uint8), np.array(color_max, dtype=np.uint8))
img_masked = cv2.bitwise_and(img, img, mask=mask_color)

# --- Chuyển sang gray, lọc nhiễu và threshold ---
gray = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# --- Loại bỏ vùng nhỏ bằng connected components ---
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
clean_binary = np.zeros_like(binary)
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] > min_area:
        clean_binary[labels == i] = 255

# --- Skeleton hóa ---
binary_bool = clean_binary > 0
skeleton = skeletonize(binary_bool).astype(np.uint8) * 255

height, width = skeleton.shape

# --- Tính degree (số neighbor trắng) cho mỗi pixel trên skeleton ---
degree = np.zeros_like(skeleton, dtype=np.uint8)
for y in range(1, height - 1):
    for x in range(1, width - 1):
        if skeleton[y, x] == 255:
            neigh = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    if skeleton[y + dy, x + dx] == 255:
                        neigh += 1
            degree[y, x] = neigh

# --- Tìm junctions (count neighbors >=3) và endpoints (degree == 1) ---
raw_junctions = []
for y in range(1, height - 1):
    for x in range(1, width - 1):
        if skeleton[y, x] == 255 and degree[y, x] >= 3:
            raw_junctions.append((x, y))

endpoints = [(x, y) for y in range(height) for x in range(width) if skeleton[y, x] == 255 and degree[y, x] == 1]

# --- Gộp các junctions gần nhau thành 1 điểm trung tâm ---
def group_points(points, radius=5):
    grouped = []
    used = [False] * len(points)
    pts = list(points)
    for i, (x1, y1) in enumerate(pts):
        if used[i]:
            continue
        group = [(x1, y1)]
        used[i] = True
        for j, (x2, y2) in enumerate(pts):
            if not used[j] and np.hypot(x1 - x2, y1 - y2) < radius:
                group.append((x2, y2))
                used[j] = True
        gx = int(np.mean([p[0] for p in group]))
        gy = int(np.mean([p[1] for p in group]))
        grouped.append((gx, gy))
    return grouped

junctions = group_points(raw_junctions, radius=group_radius)

# --- Hàm lấy neighbor trắng ---
def get_neighbors(x, y):
    nbs = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and skeleton[ny, nx] == 255:
                nbs.append((nx, ny))
    return nbs

# --- Trace từ start đến khi gặp junction/endpoint hoặc biên, đánh dấu visited ---
visited = np.zeros_like(skeleton, dtype=bool)

def trace_from(start, next_pixel):
    path = [start]
    prev = start
    cur = next_pixel
    steps = 0
    max_steps = width * height
    while True:
        path.append(cur)
        visited[cur[1], cur[0]] = True
        # dừng khi đến junction, endpoint hoặc biên
        if cur in junctions or degree[cur[1], cur[0]] == 1 or cur[0] == 0 or cur[1] == 0 or cur[0] == width - 1 or cur[1] == height - 1:
            break
        neighbors = [n for n in get_neighbors(cur[0], cur[1]) if n != prev]
        if not neighbors:
            break
        # chọn neighbor chưa visited ưu tiên
        next_choice = None
        for n in neighbors:
            if not visited[n[1], n[0]]:
                next_choice = n
                break
        if next_choice is None:
            next_choice = neighbors[0]
        prev, cur = cur, next_choice
        steps += 1
        if steps > max_steps:
            break
    return path

# --- Thu thập các segments ---
segments = []
segments_seen = set()  # lưu tuple endpoints chuẩn hoá để tránh trùng

def endpoints_key(a, b):
    return tuple(sorted([a, b]))

# 1) Trace bắt đầu từ junctions
for j in junctions:
    for nb in get_neighbors(j[0], j[1]):
        if not visited[nb[1], nb[0]]:
            seg = trace_from(j, nb)
            if len(seg) > 2:
                key = endpoints_key(seg[0], seg[-1])
                if key not in segments_seen:
                    segments_seen.add(key)
                    segments.append(seg)

# 2) Trace bắt đầu từ endpoints (đảm bảo đoạn không có junction cũng được lấy)
for e in endpoints:
    for nb in get_neighbors(e[0], e[1]):
        if not visited[nb[1], nb[0]]:
            seg = trace_from(e, nb)
            if len(seg) > 1:
                key = endpoints_key(seg[0], seg[-1])
                if key not in segments_seen:
                    segments_seen.add(key)
                    segments.append(seg)

# 3) Xử lý các pixel trắng chưa visited (vòng kín hoặc đoạn cô lập)
for y in range(height):
    for x in range(width):
        if skeleton[y, x] == 255 and not visited[y, x]:
            nbs = get_neighbors(x, y)
            if not nbs:
                visited[y, x] = True
                continue
            seg = trace_from((x, y), nbs[0])
            if len(seg) > 1:
                key = endpoints_key(seg[0], seg[-1])
                if key not in segments_seen:
                    segments_seen.add(key)
                    segments.append(seg)

print(f"Số đoạn tìm được: {len(segments)}")

# --- Lấy mẫu đều (sampled points) trên mỗi segment ---
sampled_segments = []
for seg in segments:
    sampled = []
    if len(seg) <= sample_spacing:
        sampled.append(seg[len(seg) // 2])
    else:
        for i in range(0, len(seg), sample_spacing):
            sampled.append(seg[i])
    sampled_segments.append(sampled)

# --- Tạo ảnh hiển thị: skeleton với junctions, dots, và nối lên ảnh gốc ---
# 1) Skeleton color with junctions
skeleton_color = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
for (x, y) in junctions:
    cv2.circle(skeleton_color, (x, y), 3, (0, 0, 255), -1)

# 2) Dots on skeleton
dots_image = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
for sampled in sampled_segments:
    for (x, y) in sampled:
        cv2.circle(dots_image, (x, y), 2, (0, 255, 255), -1)
for (x, y) in junctions:
    cv2.circle(dots_image, (x, y), 3, (0, 0, 255), -1)

# 3) Segments connected on clean skeleton image (colored)
colored_segments = np.zeros((*skeleton.shape, 3), dtype=np.uint8)
for seg in segments:
    color = [int(c) for c in (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))]
    for (x, y) in seg:
        colored_segments[y, x] = color
for (x, y) in junctions:
    cv2.circle(colored_segments, (x, y), 3, (0, 0, 255), -1)

# 4) Nối các điểm sampled và vẽ lên ảnh gốc
connected_on_original = img.copy()
for sampled in sampled_segments:
    color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    for i in range(len(sampled) - 1):
        pt1 = tuple(sampled[i])
        pt2 = tuple(sampled[i + 1])
        cv2.line(connected_on_original, pt1, pt2, color, thickness=2)
for (x, y) in junctions:
    cv2.circle(connected_on_original, (x, y), 6, (0, 0, 255), -1)

# --- Hiển thị / lưu ảnh kết quả ---
cv2.imshow('skeleton_junctions', skeleton_color)
cv2.imshow('skeleton_dots', dots_image)
cv2.imshow('colored_segments', colored_segments)
cv2.imshow('segments_connected_on_original', connected_on_original)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Lưu kết quả
cv2.imwrite('skeleton_junctions.jpg', skeleton_color)
cv2.imwrite('skeleton_dots.jpg', dots_image)
cv2.imwrite('colored_segments.jpg', colored_segments)
cv2.imwrite('segments_connected_on_original.jpg', connected_on_original)

# --- In tọa độ các chấm theo từng đoạn ---
print("\nTọa độ các chấm theo từng đoạn (sampled):")
for i, sampled in enumerate(sampled_segments):
    print(f"Đoạn {i+1} ({len(sampled)} chấm):", end="\n")
    for (x, y) in sampled:
        print(f"  ({x}, {y})")
