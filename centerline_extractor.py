import cv2
import numpy as np
from skimage.morphology import skeletonize
import math
from collections import defaultdict
from difflib import SequenceMatcher  


# Hàm để gom các điểm gần nhau thành cụm
def cluster_points(points, distance_threshold=10):
    points = np.array(points)
    if len(points) == 0:
        return []
    clusters = []
    visited = np.zeros(len(points), dtype=bool)

    for i, p in enumerate(points):
        if visited[i]:
            continue
        cluster_idx = np.linalg.norm(points - p, axis=1) < distance_threshold
        visited[cluster_idx] = True
        cluster_points = points[cluster_idx]
        center = cluster_points.mean(axis=0)
        clusters.append((int(center[0]), int(center[1])))

    return clusters

# Hàm để tính góc giữa hai vector
def angle_between(v1, v2):
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    norm1 = math.hypot(*v1)
    norm2 = math.hypot(*v2)
    if norm1*norm2 == 0:
        return 0
    cosang = max(-1, min(1, dot/(norm1*norm2)))
    return math.degrees(math.acos(cosang))

# Hàm để tách polyline thành các đoạn con dựa trên góc
def split_polyline_by_angle(polyline, angle_threshold=100):
    if len(polyline) < 3:
        return [polyline]
    sub_polylines = []
    current_line = [polyline[0]]
    for i in range(1, len(polyline)-1):
        p_prev = polyline[i-1]
        p_curr = polyline[i]
        p_next = polyline[i+1]
        v1 = (p_curr[0]-p_prev[0], p_curr[1]-p_prev[1])
        v2 = (p_next[0]-p_curr[0], p_next[1]-p_curr[1])
        ang = angle_between(v1,v2)
        current_line.append(p_curr)
        if ang > angle_threshold:
            sub_polylines.append(current_line)
            current_line = [p_curr]
    current_line.append(polyline[-1])
    if len(current_line) > 1:
        sub_polylines.append(current_line)
    return sub_polylines

# Hàm chính để trích xuất centerline và nút giao
def extract_centerline_and_junctions(image_path,
                                     color_min=(192,72,0),
                                     color_max=(243,242,219),
                                     epsilon_factor=0.00002,
                                     cluster_distance=15,
                                     angle_threshold=100,
                                     debug=False):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    lower = np.array(color_min, dtype=np.uint8)
    upper = np.array(color_max, dtype=np.uint8)
    mask = cv2.inRange(img, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    mask_bool = mask > 0
    skeleton = skeletonize(mask_bool)
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)

    contours, _ = cv2.findContours(skeleton_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    vis = img.copy()
    all_polylines = []
    all_vertices = []

    for cnt in contours:
        if cv2.arcLength(cnt, False) < 50:
            continue
        epsilon = 0.0003 * cv2.arcLength(cnt, False)
        approx = cv2.approxPolyDP(cnt, epsilon, False)
        polyline = [tuple(pt[0]) for pt in approx]
        all_polylines.append(polyline)
        all_vertices.extend(polyline)

    all_sub_polylines = []
    for poly in all_polylines:
        subs = split_polyline_by_angle(poly, angle_threshold=angle_threshold)
        all_sub_polylines.extend(subs)

    junction_points = cluster_points(all_vertices, distance_threshold=cluster_distance)

    # Tìm các điểm giữa ngã ba và ngã tư
    junction_midpoints = []
    for junc in junction_points:
        connected = []
        for seg in all_sub_polylines:
            for pt in (seg[0], seg[-1]):
                if np.linalg.norm(np.array(pt) - np.array(junc)) < cluster_distance:
                    connected.append(pt)
                    break
        if 3 <= len(connected) <= 4:
            center = np.mean(connected, axis=0)
            junction_midpoints.append((int(center[0]), int(center[1])))

    # Loại bỏ full duplicate segments (như cũ)
    all_endpoints = []
    for sub in all_sub_polylines:
        all_endpoints.append(sub[0])
        all_endpoints.append(sub[-1])
    endpoint_clusters = cluster_points(all_endpoints, distance_threshold=cluster_distance)
    
    # Hàm để tìm cụm gần nhất
    def get_cluster(pt):
        if not endpoint_clusters:
            return 
        dists = [np.linalg.norm(np.array(c) - np.array(pt)) for c in endpoint_clusters]
        min_idx = np.argmin(dists)
        return endpoint_clusters[min_idx]

    seg_groups = defaultdict(list)
    for sub in all_sub_polylines:
        start_cluster = get_cluster(sub[0])
        end_cluster = get_cluster(sub[-1])
        key = tuple(sorted([start_cluster, end_cluster], key=lambda x: (x[0], x[1])))
        seg_groups[key].append(sub)

    unique_sub_polylines = []
    for key, group in seg_groups.items():
        longest = max(group, key=len)
        unique_sub_polylines.append(longest)

    #  Xử lý partial overlaps
    def serialize_polyline(poly):
        return ','.join([f"{x}:{y}" for x, y in poly])
    # Hàm để tính tỉ lệ overlap giữa hai polyline
    def overlap_ratio(a, b):
        a_ser = serialize_polyline(a)
        b_ser = serialize_polyline(b)
        matcher = SequenceMatcher(None, a_ser, b_ser)
        match = matcher.find_longest_match(0, len(a_ser), 0, len(b_ser))
        return match.size / min(len(a_ser), len(b_ser)) if min(len(a_ser), len(b_ser)) > 0 else 0

    # Sắp xếp theo độ dài giảm dần để ưu tiên giữ segments dài (super-segments), loại bỏ con nếu overlap cao
    # Nếu muốn giữ atomic shorts, comment sort và thêm reverse=False hoặc sort tăng dần
    unique_sub_polylines.sort(key=len, reverse=True)

    kept_polylines = []
    for poly in unique_sub_polylines:
        if all(overlap_ratio(poly, k) < 0.5 for k in kept_polylines):  # Adjust threshold nếu cần (0.5 = 50% overlap)
            kept_polylines.append(poly)

    # Sắp xếp lại để output ổn định
    kept_polylines.sort(key=lambda sub: (sub[0][0], sub[0][1]))

    if debug:
        for i, seg in enumerate(kept_polylines):
            pts_np = np.array(seg, dtype=np.int32)
            
            # Tạo màu ngẫu nhiên cho mỗi đường
            np.random.seed(i)  # để màu ổn định giữa các lần chạy
            color = tuple(int(c) for c in np.random.randint(0, 255, 3))
            
            cv2.polylines(vis, [pts_np], isClosed=False, color=color, thickness=2)
    
        for p in junction_points:
            cv2.circle(vis, p, 3, (255, 0, 0), -1)
    
        for p in junction_midpoints:
            cv2.circle(vis, p, 6, (0, 255, 255), -1)
    
        cv2.imshow("Skeleton centerline", skeleton_uint8)
        cv2.imshow("Centerline + segments + nút giao + midpoints", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # Cập nhật junction_midpoints nếu cần (dùng kept_polylines)
    junction_midpoints = [] 
    for junc in junction_points:
        connected = []
        for seg in kept_polylines:
            for pt in (seg[0], seg[-1]):
                if np.linalg.norm(np.array(pt) - np.array(junc)) < cluster_distance:
                    connected.append(pt)
                    break
        connected = list(set(map(tuple, connected)))  
        if 3 <= len(connected) <= 4:
            center = np.mean(connected, axis=0)
            junction_midpoints.append((int(center[0]), int(center[1])))

    return kept_polylines, junction_points, junction_midpoints

