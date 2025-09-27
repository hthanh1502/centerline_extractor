import cv2
import numpy as np
import random
from skimage.morphology import skeletonize

def extract_centerline_and_junctions(image_path, debug=False):
    # --- Tham số cấu hình ---
    color_min = (192, 72, 0)
    color_max = (243, 242, 219)
    min_area = 500
    sample_spacing = 5
    group_radius = 5

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh tại: {image_path}")

    mask_color = cv2.inRange(img, np.array(color_min, dtype=np.uint8), np.array(color_max, dtype=np.uint8))
    img_masked = cv2.bitwise_and(img, img, mask=mask_color)

    gray = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean_binary = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > min_area:
            clean_binary[labels == i] = 255

    binary_bool = clean_binary > 0
    skeleton = skeletonize(binary_bool).astype(np.uint8) * 255
    height, width = skeleton.shape

    degree = np.zeros_like(skeleton, dtype=np.uint8)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if skeleton[y, x] == 255:
                degree[y, x] = cv2.countNonZero(skeleton[y-1:y+2, x-1:x+2]) - 1

    raw_junctions = [(x, y) for y in range(1, height - 1) for x in range(1, width - 1)
                     if skeleton[y, x] == 255 and degree[y, x] >= 3]
    endpoints = [(x, y) for y in range(height) for x in range(width)
                 if skeleton[y, x] == 255 and degree[y, x] == 1]

    def group_points(points, radius=5):
        grouped = []
        used = [False] * len(points)
        for i, (x1, y1) in enumerate(points):
            if used[i]: continue
            group = [(x1, y1)]
            used[i] = True
            for j, (x2, y2) in enumerate(points):
                if not used[j] and np.hypot(x1 - x2, y1 - y2) < radius:
                    group.append((x2, y2))
                    used[j] = True
            gx = int(np.mean([p[0] for p in group]))
            gy = int(np.mean([p[1] for p in group]))
            grouped.append((gx, gy))
        return grouped

    junctions = group_points(raw_junctions, radius=group_radius)

    def get_neighbors(x, y):
        return [(x+dx, y+dy) for dx in [-1,0,1] for dy in [-1,0,1]
                if (dx != 0 or dy != 0) and 0 <= x+dx < width and 0 <= y+dy < height and skeleton[y+dy, x+dx] == 255]

    visited = np.zeros_like(skeleton, dtype=bool)
    segments = []
    segments_seen = set()

    def trace_from(start, next_pixel):
        path = [start]
        prev = start
        cur = next_pixel
        while True:
            path.append(cur)
            visited[cur[1], cur[0]] = True
            if cur in junctions or degree[cur[1], cur[0]] == 1:
                break
            neighbors = [n for n in get_neighbors(cur[0], cur[1]) if n != prev and not visited[n[1], n[0]]]
            if not neighbors:
                break
            prev, cur = cur, neighbors[0]
        return path

    def endpoints_key(a, b):
        return tuple(sorted([a, b]))

    # --- Thu thập các đoạn ---
    for j in junctions + endpoints:
        for nb in get_neighbors(j[0], j[1]):
            if not visited[nb[1], nb[0]]:
                seg = trace_from(j, nb)
                key = endpoints_key(seg[0], seg[-1])
                if len(seg) > 2 and key not in segments_seen:
                    segments_seen.add(key)
                    segments.append(seg)

    for y in range(height):
        for x in range(width):
            if skeleton[y, x] == 255 and not visited[y, x]:
                nbs = get_neighbors(x, y)
                if nbs:
                    seg = trace_from((x, y), nbs[0])
                    key = endpoints_key(seg[0], seg[-1])
                    if len(seg) > 2 and key not in segments_seen:
                        segments_seen.add(key)
                        segments.append(seg)

    # --- Loại bỏ các đoạn chỉ có 1 điểm ---
    segments = [seg for seg in segments if len(seg) > 2]

    # --- Lấy mẫu đều ---
    sampled_segments = []
    for seg in segments:
        if len(seg) < 2:
            continue 
        sampled = [seg[i] for i in range(0, len(seg), sample_spacing)] if len(seg) > sample_spacing else [seg[len(seg)//2]]
        sampled_segments.append(sampled)

    # --- Tính điểm giữa các junctions ---
    junction_midpoints = []
    for j in junctions:
        neighbors = get_neighbors(j[0], j[1])
        if len(neighbors) >= 2:
            mx = int(np.mean([pt[0] for pt in neighbors]))
            my = int(np.mean([pt[1] for pt in neighbors]))
            junction_midpoints.append((mx, my))

    return sampled_segments, junctions, junction_midpoints


    # if debug:
    #     debug_img = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    #     for i, seg in enumerate(sampled_segments):
    #         pts_np = np.array(seg, dtype=np.int32)
    #         np.random.seed(i)
    #         color = tuple(int(c) for c in np.random.randint(0, 255, 3))
    #         cv2.polylines(debug_img, [pts_np], isClosed=False, color=color, thickness=1)
    #     for p in junctions:
    #         cv2.circle(debug_img, tuple(p), 4, (0, 0, 255), -1)
    #     for p in junction_midpoints:
    #         cv2.circle(debug_img, tuple(p), 6, (0, 255, 255), -1)
    #     cv2.imshow("Debug", debug_img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
