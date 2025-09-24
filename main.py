# import cv2
# import numpy as np
# from skimage.morphology import skeletonize
# import math
# import random

# def cluster_points(points, distance_threshold=10):
#     points = np.array(points)
#     if len(points) == 0:
#         return []
#     clusters = []
#     visited = np.zeros(len(points), dtype=bool)

#     for i, p in enumerate(points):
#         if visited[i]:
#             continue
#         cluster_idx = np.linalg.norm(points - p, axis=1) < distance_threshold
#         visited[cluster_idx] = True
#         cluster_points = points[cluster_idx]
#         center = cluster_points.mean(axis=0)
#         clusters.append((int(center[0]), int(center[1])))

#     return clusters

# def angle_between(v1, v2):
#     dot = v1[0]*v2[0] + v1[1]*v2[1]
#     norm1 = math.hypot(*v1)
#     norm2 = math.hypot(*v2)
#     if norm1*norm2 == 0:
#         return 0
#     cosang = max(-1, min(1, dot/(norm1*norm2)))
#     return math.degrees(math.acos(cosang))

# def split_polyline_by_angle(polyline, angle_threshold=100):
#     """
#     Tách 1 polyline thành các polyline con nếu góc đổi hướng > angle_threshold (độ).
#     """
#     if len(polyline) < 3:
#         return [polyline]  # quá ít điểm thì không chia

#     sub_polylines = []
#     current_line = [polyline[0]]

#     for i in range(1, len(polyline)-1):
#         p_prev = polyline[i-1]
#         p_curr = polyline[i]
#         p_next = polyline[i+1]

#         v1 = (p_curr[0]-p_prev[0], p_curr[1]-p_prev[1])
#         v2 = (p_next[0]-p_curr[0], p_next[1]-p_curr[1])

#         ang = angle_between(v1,v2)

#         current_line.append(p_curr)

#         if ang > angle_threshold:  # cắt tại đây
#             sub_polylines.append(current_line)
#             current_line = [p_curr]

#     # add last segment
#     current_line.append(polyline[-1])
#     if len(current_line) > 1:
#         sub_polylines.append(current_line)

#     return sub_polylines

# def extract_centerline_and_junctions(image_path,
#                                      color_min=(192,72,0),
#                                      color_max=(243,242,219),
#                                      epsilon_factor=0.000000005,
#                                      cluster_distance=15,
#                                      angle_threshold=100,
#                                      debug=False):
  
#     img = cv2.imread(image_path)
#     if img is None:
#         raise FileNotFoundError(image_path)

#     # 1. Lọc màu
#     lower = np.array(color_min, dtype=np.uint8)
#     upper = np.array(color_max, dtype=np.uint8)
#     mask = cv2.inRange(img, lower, upper)

#     # 2. Morphology để nối đường
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

#     # 3. Skeletonize để lấy centerline
#     mask_bool = mask > 0
#     skeleton = skeletonize(mask_bool)
#     skeleton_uint8 = (skeleton * 255).astype(np.uint8)

#     # 4. Tìm contour trên skeleton
#     contours, _ = cv2.findContours(skeleton_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#     vis = img.copy()
#     all_polylines = []
#     all_vertices = []

#     for cnt in contours:
#         if cv2.arcLength(cnt, False) < 50:
#             continue  

#         # 5. Xấp xỉ contour thành polyline ít điểm hơn 
#         epsilon = epsilon_factor * cv2.arcLength(cnt, False)
#         approx = cv2.approxPolyDP(cnt, epsilon, False)

#         polyline = [tuple(pt[0]) for pt in approx]

#         all_polylines.append(polyline)
#         all_vertices.extend(polyline)

#     # 6. Chia polyline thành các đoạn nhỏ khi góc đổi hướng > angle_threshold
#     all_sub_polylines = []
#     for poly in all_polylines:
#         subs = split_polyline_by_angle(poly, angle_threshold=angle_threshold)
#         all_sub_polylines.extend(subs)

#     # 7. Gom các điểm gần nhau thành nút giao (1 điểm ở giữa)
#     junction_points = cluster_points(all_vertices, distance_threshold=cluster_distance)
                                                                                                              
#     # Vẽ từng đoạn màu ngẫu nhiên
#     for seg in all_sub_polylines:
#         pts_np = np.array(seg, dtype=np.int32)
#         color = tuple(int(c) for c in np.random.randint(0,255,3))
#         cv2.polylines(vis, [pts_np], isClosed=False, color=color, thickness=2)

#     # Vẽ điểm nút giao xanh dương
#     for p in junction_points:
#         cv2.circle(vis, p, 3, (255,0,0), -1)

#     if debug:
#         cv2.imshow("Skeleton centerline", skeleton_uint8)
#         cv2.imshow("Centerline + segments + nút giao", vis)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

#     return all_sub_polylines, junction_points

# if __name__ == "__main__":
#     image_path = "img/map_image_2.jpg.png"
#     polylines, junctions = extract_centerline_and_junctions(image_path,
#                                                            epsilon_factor=0.0002,
#                                                            cluster_distance=15,
#                                                            angle_threshold=100,
#                                                            debug=True)
#     print("Tìm được", len(polylines), "đoạn centerline (sau chia).")
#     print("Tìm được", len(junctions), "nút giao.")
#     for i, poly in enumerate(polylines,1):
#         print(f"Đoạn {i}: {poly}")
#     print("Các nút giao:", junctions)

#     # Ghi ra file txt
#     with open("centerlines_segments.txt", "w", encoding="utf-8") as f:
#         f.write(f"Tìm được {len(polylines)} đoạn centerline.\n")
#         for i, poly in enumerate(polylines, 1):
#             line = f"Đoạn {i}: " + ", ".join([f"({x},{y})" for (x,y) in poly])
#             f.write(line + "\n")
#         f.write("\nCác nút giao thông:\n")
#         for j, p in enumerate(junctions,1):
#             f.write(f"Nút {j}: ({p[0]},{p[1]})\n")
#     print("Đã ghi kết quả vào file centerlines_segments.txt")
