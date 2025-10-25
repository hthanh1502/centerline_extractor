# Import các thư viện cần thiết
import cv2
import numpy as np
from skimage.morphology import skeletonize
import random

# --- GIAI ĐOẠN 1: TIỀN XỬ LÝ (PRE-PROCESSING) ---
img = cv2.imread("img/crop3.png")
if img is None:
    raise FileNotFoundError("Không tìm thấy ảnh đầu vào.")

color_min_fn = np.array([0, 0, 0], dtype=np.uint8)
color_max_fn = np.array([220, 215, 255], dtype=np.uint8)

mask_fn = cv2.inRange(img, color_min_fn, color_max_fn)

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_fn, connectivity=8)
clean_binary = np.zeros_like(mask_fn)
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] > 200:
        clean_binary[labels == i] = 255

# --- GIAI ĐOẠN 2: XƯƠNG HÓA (SKELETONIZATION) ---
binary_bool = clean_binary > 0
skeleton = skeletonize(binary_bool).astype(np.uint8) * 255
height, width = skeleton.shape

# --- GIAI ĐOẠN 3: XÁC ĐỊNH NÚT (NODE IDENTIFICATION) ---
print("Đang tìm nút giao (Cách 1: Đếm lân cận 3x3)...")
junctions = []
for y in range(1, height - 1):
    for x in range(1, width - 1):
        if skeleton[y, x] == 255:
            roi = skeleton[y-1:y+2, x-1:x+2]
            count = cv2.countNonZero(roi) - 1
            if count >= 3:
                junctions.append((x, y))

print(f"Tìm thấy {len(junctions)} điểm nút giao thô.")

# Thêm các điểm gần viền ảnh
border_margin = 5
for y in range(height):
    for x in range(width):
        if skeleton[y, x] == 255:
            if x < border_margin or y < border_margin or x >= width - border_margin or y >= height - border_margin:
                if (x, y) not in junctions:
                    junctions.append((x, y))

# Tính "degree" cho tất cả pixel trên skeleton
degree = np.zeros_like(skeleton, dtype=np.uint8)
for y in range(1, height - 1):
    for x in range(1, width - 1):
        if skeleton[y, x] == 255:
            roi = skeleton[y-1:y+2, x-1:x+2]
            degree[y, x] = cv2.countNonZero(roi) - 1

endpoints = [(x, y) for y in range(height) for x in range(width)
             if skeleton[y, x] == 255 and degree[y, x] == 1]

# --- GIAI ĐOẠN 4: TRUY VẾT CẠNH (EDGE TRACING) ---
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
            if length > 13:
                endpoints_pair = tuple(sorted([segment[0], segment[-1]]))
                if endpoints_pair not in segments_seen:
                    segments_seen.add(endpoints_pair)
                    segments.append(segment)

# --- GIAI ĐOẠN 5: TINH CHỈNH ĐỒ THỊ (GRAPH REFINEMENT) ---
def group_junctions(junctions, degree, skeleton, radius=12):
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
        minx = max(0, min(pt[0] for pt in group) - 2)
        maxx = min(skeleton.shape[1] - 1, max(pt[0] for pt in group) + 2)
        miny = max(0, min(pt[1] for pt in group) - 2)
        maxy = min(skeleton.shape[0] - 1, max(pt[1] for pt in group) + 2)
        candidates = []
        for yy in range(miny, maxy + 1):
            for xx in range(minx, maxx + 1):
                if skeleton[yy, xx] == 255:
                    candidates.append((xx, yy))
        if candidates:
            best = None
            best_key = (-1, 1e9)
            for c in candidates:
                d = int(degree[c[1], c[0]])
                dist_mean = np.mean([np.hypot(c[0] - g[0], c[1] - g[1]) for g in group])
                key = (d, -dist_mean)
                if key > best_key:
                    best_key = key
                    best = c
            grouped.append(best)
        else:
            gx = int(np.mean([pt[0] for pt in group]))
            gy = int(np.mean([pt[1] for pt in group]))
            grouped.append((gx, gy))
    return grouped

def simplify_segment(segment, epsilon=3.0):
    contour = np.array(segment, dtype=np.int32).reshape((-1, 1, 2))
    simplified = cv2.approxPolyDP(contour, epsilon, False)
    return [tuple(pt[0]) for pt in simplified]

junctions = group_junctions(junctions, degree, skeleton, radius=12)
sampled_segments = [simplify_segment(segment, epsilon=3.0) for segment in segments]

# --- SỬA ĐỔI: TINH CHỈNH JUNCTION Ở NGÃ BA ---
def fit_line_to_points(points):
    pts = np.array(points, dtype=np.float32)
    if pts.shape[0] < 2:
        return None
    vxyx0y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    return vxyx0y0.astype(float)

def get_branch_length(segment, junction):
    """Tính độ dài của segment từ junction đến điểm cuối."""
    start, end = segment[0], segment[-1]
    d_start = np.hypot(junction[0] - start[0], junction[1] - start[1])
    d_end = np.hypot(junction[0] - end[0], junction[1] - end[1])
    return min(d_start, d_end)

def refine_t_junctions(junctions, segments, fit_len=30, search_radius=12):
    refined = []
    for gx, gy in junctions:
        # Tìm các segment gần junction
        branches = []
        for seg in segments:
            if len(seg) < 3:
                continue
            s0, s1 = seg[0], seg[-1]
            d0 = np.hypot(s0[0] - gx, s0[1] - gy)
            d1 = np.hypot(s1[0] - gx, s1[1] - gy)
            if d0 <= search_radius:
                branches.append((seg[:min(len(seg), fit_len)], s0))
            elif d1 <= search_radius:
                branches.append((seg[-min(len(seg), fit_len):], s1))

        if len(branches) < 3:
            # Nếu không phải ngã ba (ít hơn 3 nhánh), giữ nguyên junction
            refined.append((gx, gy))
            continue

        # Tìm nhánh chính (dựa trên độ dài hoặc hướng)
        branch_lengths = []
        branch_lines = []
        for branch, endpoint in branches:
            length = get_branch_length(branch, (gx, gy))
            line = fit_line_to_points(branch)
            if line is not None:
                branch_lengths.append((length, branch, endpoint))
                branch_lines.append((line, branch, endpoint))

        if len(branch_lines) < 3:
            refined.append((gx, gy))
            continue

        # Chọn nhánh chính (nhánh dài nhất)
        branch_lengths.sort(reverse=True)  # Sắp xếp theo độ dài giảm dần
        main_branch = branch_lengths[0][1]  # Nhánh dài nhất
        main_endpoint = branch_lengths[0][2]

        # Tìm điểm trên nhánh chính gần junction nhất
        main_points = np.array(main_branch, dtype=np.float32)
        distances = np.hypot(main_points[:, 0] - gx, main_points[:, 1] - gy)
        closest_idx = np.argmin(distances)
        refined_junction = tuple(map(int, main_points[closest_idx]))

        refined.append(refined_junction)

    return refined

# Áp dụng tinh chỉnh cho junctions
refined_junctions = refine_t_junctions(junctions, segments, fit_len=30, search_radius=12)
junctions = refined_junctions

# Hàm tìm node gần nhất trong junctions hoặc endpoints
def find_nearest_final_node(point, nodes_arr, max_dist=20):
    if nodes_arr.size == 0:
        return point
    distances = np.sqrt(np.sum((nodes_arr - point)**2, axis=1))
    nearest_idx = np.argmin(distances)
    if distances[nearest_idx] < max_dist:
        return tuple(nodes_arr[nearest_idx])
    return point

# Kiểm tra tọa độ mask vector
print("Tọa độ các điểm trong sampled_segments (mask vector):")
for idx, seg in enumerate(sampled_segments):
    print(f"Segment {idx+1}: {seg}")

# --- GIAI ĐOẠN 6: TRỰC QUAN HÓA ---
# Tạo ảnh cho mask vector (nền trắng)
vector_mask = np.ones_like(img) * 255  # Ảnh trắng, cùng kích thước với img
vis_img = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
vis_on_original = img.copy()

all_final_nodes = junctions + endpoints
all_final_nodes_arr = np.array(all_final_nodes)

for seg in sampled_segments:
    if len(seg) < 2:
        continue
    color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    # Điều chỉnh điểm đầu và cuối
    correct_start_node = find_nearest_final_node(seg[0], all_final_nodes_arr)
    correct_end_node = find_nearest_final_node(seg[-1], all_final_nodes_arr)
    # Tạo danh sách điểm đã điều chỉnh
    corrected_seg_path = [correct_start_node] + seg[1:-1] + [correct_end_node]
    # Vẽ từng đoạn thẳng vector
    for i in range(len(corrected_seg_path) - 1):
        start = corrected_seg_path[i]
        end = corrected_seg_path[i + 1]
        cv2.line(vector_mask, start, end, color, 2)
        cv2.line(vis_img, start, end, color, 2)
        cv2.line(vis_on_original, start, end, color, 2)

# Vẽ các nút giao
for idx, (jx, jy) in enumerate(junctions):
  
    cv2.circle(vis_img, (jx, jy), 5, (0, 0, 255), -1)

    # Đánh số junctions
    cv2.putText(vis_on_original, str(idx+1), (jx+10, jy-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


# Hiển thị kết quả
cv2.imshow("Vector Mask (Segment Colors)", vector_mask)
cv2.imshow("Processed Map (Segment Colors)", vis_img)
cv2.imshow("Processed on Original (Segment Colors)", vis_on_original)
cv2.imshow("Original Mask", mask_fn)
cv2.imshow("Mask Skeleton", skeleton)
cv2.waitKey(0)
cv2.destroyAllWindows()