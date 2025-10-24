import cv2
import numpy as np
import random
import os
from skimage.morphology import skeletonize

# --- HÀM PHỤ TRỢ (HELPER FUNCTIONS) ---
# (Tất cả các hàm này được hàm chính 'extract_centerline_and_junctions' sử dụng)
# hàm lấy pixel lân cận
def get_neighbors(x, y, width, height, skeleton):
    """Tìm các pixel lân cận (8 hướng) của 1 pixel trên skeleton."""
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
#hàm truy vết segment
def trace_segment(start, next_pixel, junctions, endpoints, skeleton, visited):
    """Truy vết 1 đoạn đường (segment) từ 'start' đến nút tiếp theo."""
    height, width = skeleton.shape
    path = [start]
    stack = [(next_pixel, start)]
    
    while stack:
        (cx, cy), prev = stack.pop()
        # Ép kiểu int ở đây để đảm bảo an toàn, mặc dù đầu vào thường là int
        cx, cy = int(cx), int(cy)
        if visited[cy, cx]:
            continue
        visited[cy, cx] = True
        path.append((cx, cy))
        
        # Chuyển đổi junctions/endpoints thành set để kiểm tra nhanh hơn
        junction_set = set(junctions)
        endpoint_set = set(endpoints)

        # Điều kiện dừng
        current_point = (cx, cy)
        if current_point in junction_set or current_point in endpoint_set or cx == 0 or cy == 0 or cx == width - 1 or cy == height - 1:
            break
        # Dừng nếu quay lại 1 junction (fix của bạn)
        if current_point in junction_set and len(path) > 5:
             break

        neighbors = [n for n in get_neighbors(cx, cy, width, height, skeleton) if n != prev and not visited[n[1], n[0]]]
        for n in neighbors:
            stack.append((n, (cx, cy))) # n đã là tuple (int, int) từ get_neighbors
    # Đảm bảo path trả về là list of tuples (int, int)
    return [(int(p[0]), int(p[1])) for p in path]

#hàm gộp nút giao
def group_junctions(junctions, degree, skeleton, radius=12):
    """Hàm gộp các nút giao nâng cao (chọn pixel đại diện tốt nhất)."""
    grouped = []
    used = [False] * len(junctions)
    # Đảm bảo junctions đầu vào là list of tuples (int, int)
    junctions_int = [(int(p[0]), int(p[1])) for p in junctions]
    
    for i, (x1, y1) in enumerate(junctions_int):
        if used[i]:
            continue
        group = [(x1, y1)]
        used[i] = True
        for j, (x2, y2) in enumerate(junctions_int):
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
                    # Ép kiểu int khi thêm candidate
                    candidates.append((int(xx), int(yy)))
        if candidates:
            best = None
            best_key = (-1, 1e9)
            for c in candidates: # c đã là (int, int)
                # Đảm bảo truy cập degree bằng index int
                d = int(degree[c[1], c[0]])
                dist_mean = np.mean([np.hypot(c[0] - g[0], c[1] - g[1]) for g in group])
                key = (d, -dist_mean)
                if key > best_key:
                    best_key = key
                    best = c
            # Ép kiểu sang int chuẩn của Python
            grouped.append((int(best[0]), int(best[1])))
        else:
            gx = int(np.mean([pt[0] for pt in group]))
            gy = int(np.mean([pt[1] for pt in group]))
            # Ép kiểu sang int chuẩn của Python
            grouped.append((int(gx), int(gy)))
    return grouped
# hàm làm mượt/đơn giản hóa segment
def simplify_segment(segment, epsilon=3.0):
    """Hàm làm mượt/đơn giản hóa segment (dùng cv2.approxPolyDP)."""
    if not segment: # Tránh lỗi nếu segment rỗng
        return []
    # Đảm bảo segment đầu vào là list of tuples (int, int)
    segment_int = [(int(p[0]), int(p[1])) for p in segment]
    contour = np.array(segment_int, dtype=np.int32).reshape((-1, 1, 2))
    simplified = cv2.approxPolyDP(contour, epsilon, False)
    # Ép kiểu từng tọa độ sang int chuẩn của Python
    return [(int(pt[0][0]), int(pt[0][1])) for pt in simplified]
#hàm fit line để tìm giao điểm
def fit_line_to_points(points):
    """Fit một đường thẳng (toán học) vào một tập hợp điểm."""
    # Đảm bảo points là list of tuples (int, int) trước khi chuyển sang float32
    points_int = [(int(p[0]), int(p[1])) for p in points]
    pts = np.array(points_int, dtype=np.float32)
    if pts.shape[0] < 2:
        return None
    vxyx0y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    return vxyx0y0.astype(float) 
#hàm tìm giao điểm 2 đường thẳng
def intersect_lines(line1, line2, eps=1e-8):
    """Tìm giao điểm của 2 đường thẳng."""
    v = line1[0:2]; p = line1[2:4]
    w = line2[0:2]; q = line2[2:4]
    A = np.column_stack((v, -w)) 
    b = (q - p)
    det = np.linalg.det(A)
    if abs(det) < eps:
        return None
    ts = np.linalg.solve(A, b)
    t = ts[0]
    inter = p + t * v
    # Kết quả intersect đã là float
    return (float(inter[0]), float(inter[1]))
#hàm tinh chỉnh nút giao bằng fit line
def refine_junctions_with_line_fitting(grouped_junctions, segments, fit_len=30, search_radius=12):
    """Tinh chỉnh vị trí nút giao bằng cách tìm giao điểm của các đường thẳng fit."""
    refined = []
    # Đảm bảo grouped_junctions là list of tuples (int, int)
    grouped_junctions_int = [(int(p[0]), int(p[1])) for p in grouped_junctions]
    
    for gx, gy in grouped_junctions_int: # gx, gy đã là int
        branches = []
        for seg in segments: # seg là list of tuples (int, int) từ trace_segment
            if len(seg) < 3:
                continue
            s0 = seg[0]; s1 = seg[-1] # s0, s1 là (int, int)
            d0 = np.hypot(s0[0] - gx, s0[1] - gy)
            d1 = np.hypot(s1[0] - gx, s1[1] - gy)
            if d0 <= search_radius:
                pts = seg[:min(len(seg), fit_len)] # pts là list of tuples (int, int)
                branches.append(pts)
            elif d1 <= search_radius:
                pts = seg[-min(len(seg), fit_len):] # pts là list of tuples (int, int)
                branches.append(pts)
                
        if len(branches) < 2:
            refined.append((gx, gy)) # Giữ nguyên nút giao cũ (đã là int)
            continue
            
        lines = []
        for pts in branches: # pts đã là list of tuples (int, int)
            line = fit_line_to_points(pts) # fit_line cần int input
            if line is not None:
                lines.append(line)
                
        if len(lines) < 2:
            refined.append((gx, gy)) # Giữ nguyên nút giao cũ (đã là int)
            continue
            
        intersections = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                inter = intersect_lines(lines[i], lines[j]) # inter là (float, float) hoặc None
                if inter is not None:
                    intersections.append(inter)
                    
        if not intersections:
            refined.append((gx, gy)) # Giữ nguyên nút giao cũ (đã là int)
            continue
            
        dists = [np.hypot(x - gx, y - gy) for x, y in intersections]
        min_idx = int(np.argmin(dists))
        chosen = intersections[min_idx] # chosen là (float, float)
        # Ép kiểu sang int chuẩn của Python khi thêm vào kết quả
        refined.append((int(round(chosen[0])), int(round(chosen[1]))))
    return refined
#hàm tìm nút cuối gần nhất
def find_nearest_final_node(point, nodes_arr, max_dist=20):
    """Tìm Nút cuối cùng (từ all_final_nodes) gần 'point' nhất."""
    # Đảm bảo point là (int, int)
    point_int = (int(point[0]), int(point[1]))
    if nodes_arr.size == 0:
        return point_int 
    # nodes_arr nên được đảm bảo là array of (int, int) trước khi gọi hàm này
    distances = np.sqrt(np.sum((nodes_arr - point_int)**2, axis=1))
    nearest_idx = np.argmin(distances)
    
    if distances[nearest_idx] < max_dist:
        nearest_point = nodes_arr[nearest_idx]
        # Ép kiểu kết quả (từ mảng NumPy) sang int chuẩn của Python
        return (int(nearest_point[0]), int(nearest_point[1]))
    else:
        return point_int
#hàm tính góc giữa 2 vector
def get_angle_between_vectors(v1, v2):
    """Tính góc (radian) giữa 2 vector."""
    # Đảm bảo vector là float để chuẩn hóa
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0: return 0 # Tránh chia cho 0
    v1_u = v1 / norm_v1
    v2_u = v2 / norm_v2
    dot_product = np.dot(v1_u, v2_u)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    return np.arccos(dot_product)
#hàm kiểm tra 3 điểm có gần thẳng hàng không
def check_collinear(p1, center, p2, angle_threshold_deg=15):
    """Kiểm tra xem p1, center, p2 có gần thẳng hàng không (góc ~180 độ)."""
    # Đảm bảo các điểm là (int, int) trước khi tạo vector
    p1_int = (int(p1[0]), int(p1[1]))
    center_int = (int(center[0]), int(center[1]))
    p2_int = (int(p2[0]), int(p2[1]))
    v1 = np.array(p1_int) - np.array(center_int)
    v2 = np.array(p2_int) - np.array(center_int)
    
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return False
    
    angle_rad = get_angle_between_vectors(v1, v2)
    angle_deg = np.degrees(angle_rad)
    
    return abs(angle_deg - 180) < angle_threshold_deg

# --- HÀM XỬ LÝ CHÍNH (ĐƯỢC GỌI BỞI APP.PY) ---
#MARK: mask
def extract_centerline_and_junctions(image_path, debug=False):
    """
    Hàm chính: Chạy toàn bộ quy trình từ Giai đoạn 1 đến 5.6.
    Sử dụng tiền xử lý inpaint MỚI và tinh chỉnh NÂNG CAO.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh tại: {image_path}")

    # --- GIAI ĐOẠN 1: TIỀN XỬ LÝ (TỪ CODE MỚI CỦA BẠN) ---
    print("Giai đoạn 1: Đang tiền xử lý (Inpaint, Bridging)...")
    
    color_min_tong = np.array([0, 0, 0], dtype=np.uint8)
    color_max_tong = np.array([220, 215, 255], dtype=np.uint8)
    color_min_duong = np.array([170, 0, 0], dtype=np.uint8)
    color_max_duong = np.array([255, 229, 255], dtype=np.uint8)
    color_min_chu = np.array([0, 0, 0], dtype=np.uint8)
    color_max_chu = np.array([255, 180, 255], dtype=np.uint8)
    color_min_cam = np.array([0, 100, 200], dtype=np.uint8)
    color_max_cam = np.array([80, 180, 255], dtype=np.uint8)
    
    mask_cam = cv2.inRange(img, color_min_cam, color_max_cam)
    mask_chu = cv2.inRange(img, color_min_chu, color_max_chu)

    kernel_chu = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_chu = cv2.dilate(mask_chu, kernel_chu, iterations=3)
    mask_cam = cv2.dilate(mask_cam, kernel_chu, iterations=2)
    mask_remove = cv2.bitwise_or(mask_chu, mask_cam)

    img_inpainted = cv2.inpaint(img, mask_remove, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    mask_duong_tong = cv2.inRange(img_inpainted, color_min_tong, color_max_tong)
    mask_duong_raw = cv2.inRange(img_inpainted, color_min_duong, color_max_duong)
    mask_road_clean = cv2.bitwise_or(mask_duong_tong, mask_duong_raw)

    kernel_noise = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_road_clean = cv2.morphologyEx(mask_road_clean, cv2.MORPH_OPEN, kernel_noise)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    mask_road_clean = cv2.morphologyEx(mask_road_clean, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    dist = cv2.distanceTransform(mask_road_clean, cv2.DIST_L2, 5)
    _, dist_thresh = cv2.threshold(dist, 8, 255, cv2.THRESH_BINARY)
    dist_thresh = dist_thresh.astype(np.uint8)
    kernel_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))
    mask_bridge = cv2.dilate(dist_thresh, kernel_bridge, iterations=1)
    mask_road_clean = cv2.bitwise_or(mask_road_clean, mask_bridge)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_duong_tong, connectivity=8)
    clean_binary = np.zeros_like(mask_duong_tong)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 200:
            clean_binary[labels == i] = 255

    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_binary = cv2.morphologyEx(clean_binary, cv2.MORPH_CLOSE, kernel_smooth)

    # --- GIAI ĐOẠN 2: XƯƠNG HÓA ---
    print("Giai đoạn 2: Đang xương hóa...")
    binary_bool = clean_binary > 0
    skeleton = skeletonize(binary_bool).astype(np.uint8) * 255
    height, width = skeleton.shape

    # --- GIAI ĐOẠN 3: XÁC ĐỊNH NÚT THÔ ---
    print("Giai đoạn 3: Đang tìm nút thô (Junctions, Endpoints)...")
    junctions_raw = [] # Đổi tên để tránh nhầm lẫn
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if skeleton[y, x] == 255:
                roi = skeleton[y-1:y+2, x-1:x+2]
                count = cv2.countNonZero(roi) - 1
                if count >= 3:
                    junctions_raw.append((int(x), int(y))) # Ép kiểu int

    border_margin = 5
    for y in range(height):
        for x in range(width):
            if skeleton[y, x] == 255:
                # Ép kiểu int khi thêm vào
                point_int = (int(x), int(y))
                if (x < border_margin or y < border_margin or x >= width - border_margin or y >= height - border_margin) and point_int not in junctions_raw:
                     junctions_raw.append(point_int)

    degree = np.zeros_like(skeleton, dtype=np.uint8)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if skeleton[y, x] == 255:
                roi = skeleton[y-1:y+2, x-1:x+2]
                degree[y, x] = cv2.countNonZero(roi) - 1

    endpoints_raw = [(int(x), int(y)) for y in range(height) for x in range(width) # Ép kiểu int
                 if skeleton[y, x] == 255 and degree[y, x] == 1]

    # --- GIAI ĐOẠN 4: TRUY VẾT CẠNH (SEGMENTS) ---
    print("Giai đoạn 4: Đang truy vết các đoạn đường...")
    visited = np.zeros_like(skeleton, dtype=bool)
    segments_raw = [] # Đổi tên
    segments_seen = set()
    
    # Chuyển đổi junctions/endpoints sang set để kiểm tra nhanh hơn trong trace_segment
    junction_set_raw = set(junctions_raw)
    endpoint_set_raw = set(endpoints_raw)

    for start_node in junctions_raw + endpoints_raw: # start_node đã là (int, int)
        for neighbor in get_neighbors(start_node[0], start_node[1], width, height, skeleton): # neighbor là (int, int)
            if not visited[neighbor[1], neighbor[0]]:
                # Truyền set vào để tăng tốc độ kiểm tra 'in'
                segment = trace_segment(start_node, neighbor, junction_set_raw, endpoint_set_raw, skeleton, visited)
                # segment trả về list of tuples (int, int)
                if len(segment) < 2: continue # Bỏ qua nếu segment không hợp lệ
                start, end = segment[0], segment[-1] # start, end là (int, int)
                length = np.hypot(end[0] - start[0], end[1] - start[1])
                if length > 13:
                    endpoints_pair = tuple(sorted([start, end])) # Dùng start, end đã là int
                    if endpoints_pair not in segments_seen:
                        segments_seen.add(endpoints_pair)
                        segments_raw.append(segment) # segment đã là list of tuples (int, int)

    # --- GIAI ĐOẠN 5: TINH CHỈNH ĐỒ THỊ (NÂNG CAO) ---
    print("Giai đoạn 5: Đang tinh chỉnh đồ thị...")
    
    # 5a. Gộp nút
    # group_junctions cần degree, junctions_raw (list of tuples (int, int))
    junctions_grouped = group_junctions(junctions_raw, degree, skeleton, radius=12) 
    # junctions_grouped trả về list of tuples (int, int)
    
    # 5b. Làm mượt cạnh
    sampled_segments = []
    for segment in segments_raw: # segment là list of tuples (int, int)
        # simplify_segment cần list of tuples (int, int), trả về list of tuples (int, int)
        simplified = simplify_segment(segment, epsilon=3.0) 
        sampled_segments.append(simplified)
    
    # Sắp xếp lại (không bắt buộc)
    # junction_midpoints = [tuple(seg[len(seg)//2]) for seg in sampled_segments if len(seg) >= 2]
    # sorted_data = sorted(zip(sampled_segments, junction_midpoints), key=lambda item: (item[1][1], item[1][0]))
    # sampled_segments, _ = zip(*sorted_data) if sorted_data else ([], [])
    # sampled_segments = list(sampled_segments) # sampled_segments là list of list of tuples (int, int)
    
    # 5c. Tinh chỉnh nút (Refine)
    # refine_junctions cần grouped junctions (int, int) và segments_raw (int, int)
    final_junctions = refine_junctions_with_line_fitting(junctions_grouped, segments_raw, fit_len=30, search_radius=12) 
    # final_junctions trả về list of tuples (int, int)

    # Lấy lại danh sách endpoints (không thay đổi)
    final_endpoints = endpoints_raw # Đã là list of tuples (int, int)

    # 5.5. Hòa tan (Dissolve)
    print("Giai đoạn 5.5: Đang hòa tan các nút giao thẳng...")
    # Tạo NumPy array từ list các tuple (int, int)
    all_final_nodes_arr_for_merge = np.array(final_junctions + final_endpoints, dtype=int)
    final_segment_map = {}
    if all_final_nodes_arr_for_merge.size > 0:
        for i, seg in enumerate(sampled_segments): # sampled_segments là list of list of tuples (int, int)
            if len(seg) < 2: continue
            # find_nearest cần point (int, int) và array (int, int), trả về (int, int)
            node_A = find_nearest_final_node(seg[0], all_final_nodes_arr_for_merge)
            node_B = find_nearest_final_node(seg[-1], all_final_nodes_arr_for_merge)
            # Lưu seg (list of tuples (int, int)) vào map
            final_segment_map[i] = (node_A, node_B, seg)

    node_to_segments_map = {}
    for i, (node_A, node_B, seg) in final_segment_map.items(): # node_A, node_B là (int, int)
        if node_A not in node_to_segments_map: node_to_segments_map[node_A] = []
        if node_B not in node_to_segments_map: node_to_segments_map[node_B] = []
        node_to_segments_map[node_A].append((node_B, i))
        node_to_segments_map[node_B].append((node_A, i))

    dissolved_junctions = set()
    segments_to_skip = set()
    new_merged_segments = []
    for junction in final_junctions: # junction là (int, int)
        connected = node_to_segments_map.get(junction)
        if connected and len(connected) == 2:
            neighbor_A, seg_idx_A = connected[0] # neighbor_A là (int, int)
            neighbor_B, seg_idx_B = connected[1] # neighbor_B là (int, int)
            seg_A = final_segment_map[seg_idx_A][2] # list of tuples (int, int)
            seg_B = final_segment_map[seg_idx_B][2] # list of tuples (int, int)
            if len(seg_A) == 2 and len(seg_B) == 2:
                # check_collinear cần các tuple (int, int)
                if check_collinear(neighbor_A, junction, neighbor_B, angle_threshold_deg=15):
                    dissolved_junctions.add(junction)
                    segments_to_skip.add(seg_idx_A)
                    segments_to_skip.add(seg_idx_B)
                    # Tạo segment mới chỉ gồm 2 điểm (int, int)
                    new_merged_segments.append([neighbor_A, neighbor_B])

    final_sampled_segments = new_merged_segments # list of list [ (int,int), (int,int) ]
    for i, (node_A, node_B, seg) in final_segment_map.items(): # seg là list of tuples (int, int)
        if i not in segments_to_skip:
            final_sampled_segments.append(seg) # Thêm list of tuples (int, int)
    
    # Lọc lại final_junctions
    final_junctions_dissolved = [j for j in final_junctions if j not in dissolved_junctions] 
    # final_junctions_dissolved là list of tuples (int, int)

    print(f"Đã hòa tan {len(dissolved_junctions)} nút. Số segment ban đầu: {len(sampled_segments)} -> Sau khi hòa tan: {len(final_sampled_segments)}")

    # 5.6. Chuẩn bị tọa độ để trả về (đã ép kiểu int hết ở các bước trước)
    segment_coordinates_data = []
    for i, seg in enumerate(final_sampled_segments): # seg là list of tuples (int, int)
        seg_data = {"id": i, "type": "straight" if len(seg) == 2 else "curve", "points": seg}
        segment_coordinates_data.append(seg_data)

    print("Giai đoạn 5.6: Đã trích xuất tọa độ.")
    
    # Hiển thị ảnh debug nếu được yêu cầu
    if debug:
        cv2.imshow("Mask Remove (Chu + Cam)", mask_remove)
        cv2.imshow("Img Inpainted", img_inpainted)
        cv2.imshow("Mask Road Clean", mask_road_clean)
        cv2.imshow("Clean Binary", clean_binary)
        cv2.imshow("Skeleton", skeleton)
        
        vis_skeleton = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
        # Vẽ các nút giao đã tinh chỉnh (trước khi hòa tan)
        for (jx, jy) in final_junctions: 
             cv2.circle(vis_skeleton, (jx, jy), 5, (0, 0, 255), -1) 
        # Vẽ các nút giao cuối cùng (sau khi hòa tan) màu xanh lá
        for (jx, jy) in final_junctions_dissolved: 
             cv2.circle(vis_skeleton, (jx, jy), 3, (0, 255, 0), -1) 
        for (ex, ey) in final_endpoints:
             cv2.circle(vis_skeleton, (ex, ey), 3, (0, 255, 255), -1)
        cv2.imshow("Skeleton with Final Nodes (Red=Refined, Green=Dissolved)", vis_skeleton)
        
        print("\nĐang ở chế độ Debug. Nhấn phím bất kỳ trên cửa sổ ảnh để tiếp tục...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Trả về kết quả cuối cùng để app.py sử dụng
    # Đảm bảo tất cả đều là kiểu Python chuẩn
    return final_sampled_segments, final_junctions_dissolved, []
