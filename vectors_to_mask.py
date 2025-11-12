import cv2
import numpy as np
from skimage.morphology import skeletonize
import networkx as nx
from shapely.geometry import LineString, Point, MultiPoint
import json
from sklearn.cluster import DBSCAN

# =============================================================================
# CÁC HÀM TỪ FILE 1 (PHÁT HIỆN VÀ LƯU FILE)
# =============================================================================

# --- 1. Hàm xử lý skeleton ---
def mask_to_skeleton(mask):
    skeleton = skeletonize(mask > 0).astype(np.uint8)
    return skeleton * 255

def skeleton_to_graph(skel):
    G = nx.Graph()
    h, w = skel.shape
    coords = np.argwhere(skel > 0)
    for y, x in coords:
        G.add_node((x, y))
    for y, x in coords:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx_, ny_ = x + dx, y + dy
                if 0 <= nx_ < w and 0 <= ny_ < h and skel[ny_, nx_] > 0:
                    G.add_edge((x, y), (nx_, ny_))
    return G

def graph_to_vectors(G):
    def is_junction(node):
        return G.degree(node) != 2
    visited = set()
    vectors = []
    for node in G.nodes:
        if is_junction(node):
            for neighbor in G.neighbors(node):
                path = [node]
                prev, curr = node, neighbor
                while True:
                    path.append(curr)
                    visited.add(curr)
                    next_nodes = [n for n in G.neighbors(curr) if n != prev]
                    if len(next_nodes) != 1:
                        break
                    prev, curr = curr, next_nodes[0]
                if len(path) > 2:
                    vectors.append(np.array(path, dtype=np.float32))
    return vectors

def crop_polyline(vectors, cut_length=10):
    cropped_vectors = []
    for v in vectors:
        if len(v) < 2:
            continue
        line = LineString(v)
        if line.length <= 2 * cut_length:
            continue
        coords = np.linspace(cut_length, line.length - cut_length, num=len(v))
        trimmed = [line.interpolate(d).coords[0] for d in coords]
        cropped_vectors.append(np.array(trimmed, dtype=np.float32))
    return cropped_vectors

def detect_road_vectors(image_path, color_min, color_max):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.inRange(img_rgb, color_min, color_max)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    skeleton = mask_to_skeleton(mask)
    G = skeleton_to_graph(skeleton)
    vectors = graph_to_vectors(G)
    vectors = crop_polyline(vectors, cut_length=10)
    return img, mask, skeleton, vectors
#MARK: find corners
def find_corner_points(v, angle_threshold_deg=15, min_dist=10):
    v_int = v.astype(np.int32)
    approx = cv2.approxPolyDP(v_int, epsilon=4.5, closed=False)
    approx = approx.reshape(-1, 2)
    corners = []

    for i in range(1, len(approx) - 1):
        a, b, c = approx[i - 1], approx[i], approx[i + 1]
        v1 = a - b
        v2 = c - b
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 < min_dist or norm2 < min_dist:
            continue
        cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        if angle > angle_threshold_deg:
            corners.append(b)
    return corners

def draw_vectors_and_points(img, vectors, angle_threshold=45):
    overlay = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for v in vectors:
        if len(v) < 2:
            continue

        # Vẽ polyline đỏ
        # cv2.polylines(overlay, [v.astype(np.int32)], False, (0, 0, 255), 2)

        # Lấy điểm đầu, cuối và góc gấp khúc
        start, end = v[0], v[-1]
        corners = find_corner_points(v, angle_threshold)

        # Gom các điểm đặc trưng (loại trùng lặp gần nhau)
        keypoints = [start] + corners + [end]
        keypoints_filtered = []
        for pt in keypoints:
            if not keypoints_filtered or np.linalg.norm(np.array(pt) - np.array(keypoints_filtered[-1])) > 10:
                keypoints_filtered.append(pt)

        # Vẽ các điểm đặc trưng vàng + tọa độ
        for pt in keypoints_filtered:
            pt_i = tuple(np.int32(pt))
            cv2.circle(overlay, pt_i, 3, (255, 0, 0), -1)
            cv2.putText(
                overlay,
                f"({pt_i[0]}, {pt_i[1]})",
                (pt_i[0] + 5, pt_i[1] - 5),
                font,
                0.4,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    blended = cv2.addWeighted(overlay, 0.75, img, 0.25, 0)
    return blended

def save_vectors_to_txt(vectors, filename="segments.txt"):
    data = []
    for v in vectors:
        start = v[0].tolist()
        end = v[-1].tolist()
        corners = [pt.tolist() for pt in find_corner_points(v)]
        data.append({
            "start": start,
            "corners": corners,
            "end": end,
            "points": v.tolist()
        })

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Đã lưu {len(data)} đoạn (bao gồm góc gấp khúc) vào file '{filename}'")


# =============================================================================
# CÁC HÀM TỪ FILE 2 (ĐỌC FILE VÀ HIỂN THỊ)
# =============================================================================

# --- 1. Cấu hình đường dẫn ---
# (Đã di chuyển vào phần __main__ để dùng chung)
#MARK: view segments
# --- 2. Hàm hiển thị dữ liệu ---
def view_segments(image_path, segment_file):
    # Đọc ảnh nền
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh tại {image_path}")

    # Đọc dữ liệu segment
    with open(segment_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    overlay = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    # --- Vẽ từng đoạn ---
    for i, seg in enumerate(data):
        points = np.array(seg["points"], dtype=np.float32)
        corners = [np.array(c, dtype=np.float32) for c in seg["corners"]]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # --- Gom tất cả điểm đặc trưng: start + corners + end ---
        keypoints = [np.array(seg["start"], dtype=np.float32)] + corners + [np.array(seg["end"], dtype=np.float32)]

        # --- Vẽ vector giữa các điểm liền kề ---
        for j in range(len(keypoints) - 1):
            p1 = keypoints[j]
            p2 = keypoints[j + 1]
            direction = p2 - p1
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                continue
            direction = direction / norm
            extended_end = p2 + direction * 25.0 # Kéo dài thêm 25 pixel

            p1_i = tuple(np.int32(p1))
            p2_i = tuple(np.int32(extended_end))

            # Vẽ vector (mũi tên vàng)
            cv2.line(overlay, p1_i, p2_i, (0, 0, 255), 1)
            # Hiển thị toạ độ 2 đầu
            # cv2.putText(overlay, f"({int(p1_i[0])},{int(p1_i[1])})", (p1_i[0] + 5, p1_i[1] - 5),
            #             font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
            # cv2.putText(overlay, f"({int(p2_i[0])},{int(p2_i[1])})", (p2_i[0] + 5, p2_i[1] - 5),
            #             font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

        # --- Vẽ lại điểm gấp khúc (xanh lá) ---
        for j, c in enumerate(corners):
            c_i = tuple(np.int32(c))
            cv2.circle(overlay, c_i, 3, (255, 0, 0), -1)
            # cv2.putText(overlay, f"C{i}-{j}", (c_i[0] + 3, c_i[1] - 3),
            #             font, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
    

    blended = cv2.addWeighted(overlay, 0.75, img, 0.25, 0)

    
    # --- Tạo mask đen trắng ---
    mask_vectors = np.zeros(img.shape[:2], dtype=np.uint8)  # mask đơn kênh, nền đen

    for i, seg in enumerate(data):
        points = np.array(seg["points"], dtype=np.float32)
        corners = [np.array(c, dtype=np.float32) for c in seg["corners"]]
        keypoints = [np.array(seg["start"], dtype=np.float32)] + corners + [np.array(seg["end"], dtype=np.float32)]

        for j in range(len(keypoints) - 1):
            p1 = keypoints[j]
            p2 = keypoints[j + 1]
            direction = p2 - p1
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                continue
            direction = direction / norm
            extended_end = p2 + direction * 20.0  # kéo dài

            p1_i = tuple(np.int32(p1))
            p2_i = tuple(np.int32(extended_end))

            # vẽ vector trắng trên nền đen
            cv2.line(mask_vectors, p1_i, p2_i, 255, 1)


    # --- Tìm junction bằng Shapely ---
    # --- Tìm các điểm giao nhau giữa các vector ---
    # (Đã import shapely ở đầu file)

    lines = []
    for i, seg in enumerate(data):
        corners = [np.array(c, dtype=np.float32) for c in seg["corners"]]
        keypoints = [np.array(seg["start"], dtype=np.float32)] + corners + [np.array(seg["end"], dtype=np.float32)]
        for j in range(len(keypoints) - 1):
            p1 = keypoints[j]
            p2 = keypoints[j + 1]
            direction = p2 - p1
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                continue
            direction = direction / norm
            extended_end = p2 + direction * 20.0
            line = LineString([tuple(p1), tuple(extended_end)])
            lines.append(line)

    # --- Tìm tất cả giao điểm ---
    junctions = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            inter = lines[i].intersection(lines[j])
            if inter.is_empty:
                continue
            if inter.geom_type == "Point":
                junctions.append((int(inter.x), int(inter.y)))
            elif inter.geom_type == "MultiPoint":
                for p in inter.geoms:
                    junctions.append((int(p.x), int(p.y)))

    # --- Vẽ các junction lên bản sao ảnh gốc ---
    # --- Gom các điểm junction gần nhau (DBSCAN clustering) ---
    if junctions:
        pts = np.array(junctions)
        clustering = DBSCAN(eps=10, min_samples=1).fit(pts)  # eps=5: bán kính 5 pixel

        merged_junctions = []
        for label in np.unique(clustering.labels_):
            cluster_pts = pts[clustering.labels_ == label]
            centroid = np.mean(cluster_pts, axis=0)
            merged_junctions.append(tuple(np.int32(centroid)))

        junctions = merged_junctions  # cập nhật danh sách junction sau khi gộp
        print(f"✅ Gom còn {len(junctions)} junction sau khi hợp nhất các điểm gần nhau.")
    else:
        print("⚠️ Không có junction nào được phát hiện để gom nhóm.")

    # --- Vẽ các junction đã gộp ---
    mask_junctions = img.copy()
    for pt in junctions:
        cv2.circle(mask_junctions, pt, 3, (0, 0, 255), -1)

    
    # --- Bổ sung thêm điểm đầu và cuối vào junctions ---
    for seg in data:
        start_pt = tuple(map(int, seg["start"]))
        end_pt = tuple(map(int, seg["end"]))
        junctions.append(start_pt)
        junctions.append(end_pt)

    # --- Xây dựng các đoạn nối giữa các junction thực ---
    new_segments = []

    for line in lines:
        # lấy tất cả junction nằm trên line này (kể cả start/end)
        junc_on_line = []
        for pt in junctions:
            p = Point(pt)
            if line.distance(p) < 1.5:
                junc_on_line.append(pt)

        # sắp xếp theo thứ tự dọc line
        if len(junc_on_line) >= 2:
            junc_on_line = sorted(junc_on_line, key=lambda p: line.project(Point(p)))
            for i in range(len(junc_on_line) - 1):
                p1 = junc_on_line[i]
                p2 = junc_on_line[i + 1]
                if p1 != p2 :
                    cv2.line(mask_junctions, p1, p2, (255, 255, 255), 1)
                    new_segments.append({"start": p1, "end": p2})

    print(f"✅ Tổng số đoạn đường mới: {len(new_segments)}")

    # --- Lưu kết quả ra file ---
    # output_file = "aaaaajunction_segments.txt"
    # with open(output_file, "w", encoding="utf-8") as f:
    #     json.dump(new_segments, f, indent=2, ensure_ascii=False)
    # print(f"💾 Đã lưu kết quả vào {output_file}")

    # --- Hiển thị cửa sổ ---
    cv2.imshow("View Segments", blended)
    cv2.imshow("Mask Vectors (Black & White)", mask_vectors)
    cv2.imshow("Junctions on Original", mask_junctions)


    cv2.imwrite("output/mask_junctions.png", mask_junctions)
    cv2.imwrite("output/blended.png", blended)
    cv2.imwrite("output/mask_vectors.png", mask_vectors)

    # Không gọi waitKey/destroyAllWindows ở đây
    # để hàm main bên dưới kiểm soát
    
    # Trả về các ảnh đã xử lý (để hàm main hiển thị)
    return blended, mask_vectors, mask_junctions


# =============================================================================
# PHẦN THỰC THI CHÍNH (MAIN)
# =============================================================================
#MARK: main
if __name__ == "__main__":
    
    # --- Cấu hình chung ---
    IMAGE_PATH = "img/merged_map.png"
    SEGMENTS_FILE = "segments.txt" # File trung gian

    # --- 1. Thực thi logic FILE 1 (Phát hiện và LƯU file) ---
    print("--- BẮT ĐẦU LOGIC FILE 1: PHÁT HIỆN & LƯU ---")
    color_min_fn = np.array([0, 0, 0], dtype=np.uint8)
    color_max_fn = np.array([220, 215, 255], dtype=np.uint8)

    img1, mask1, skeleton1, vectors1 = detect_road_vectors(IMAGE_PATH, color_min_fn, color_max_fn)
    result1 = draw_vectors_and_points(img1, vectors1)

    print(f"Tổng số đoạn: {len(vectors1)}")
    # for i, v in enumerate(vectors1):
    #     print(f"Đoạn {i+1}: start={v[0]}, end={v[-1]}")

    # ✅ Lưu kết quả ra file TXT dạng mảng
    save_vectors_to_txt(vectors1, SEGMENTS_FILE)

    # --- Hiển thị kết quả (tạm thời) của File 1 ---
    cv2.imshow("File 1 - Mask", mask1)
    cv2.imshow("File 1 - Skeleton", skeleton1)
    cv2.imshow("File 1 - Overlay", result1)
    # cv2.imwrite("aaoutput_overlay.png", result1)
    
    print("--- KẾT THÚC LOGIC FILE 1 ---")
    print("\n--- BẮT ĐẦU LOGIC FILE 2: ĐỌC FILE & XỬ LÝ JUNCTION ---")

    # --- 2. Thực thi logic FILE 2 (ĐỌC file và xử lý) ---
    try:
        # Gọi hàm xử lý của File 2
        # Hàm này sẽ tự đọc file, xử lý và mở cửa sổ
        view_segments(IMAGE_PATH, SEGMENTS_FILE)
        
        print("--- KẾT THÚC LOGIC FILE 2 ---")
        
        # Giữ tất cả cửa sổ mở
        print("\nNhấn phím bất kỳ trên một cửa sổ ảnh để thoát...")
        cv2.waitKey(0)
        
    except FileNotFoundError as e:
        print(f"LỖI: Không thể chạy logic File 2. {e}")
    except json.JSONDecodeError:
        print(f"LỖI: File '{SEGMENTS_FILE}' rỗng hoặc bị lỗi. Không thể chạy logic File 2.")
    except Exception as e:
        print(f"LỖI không xác định khi chạy logic File 2: {e}")

    finally:
        # Đóng tất cả cửa sổ
        cv2.destroyAllWindows()