import cv2
import numpy as np

# --- Đọc ảnh ---
image_path = "output/mask_road_clean.png"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# --- Nhị phân hóa (nếu ảnh chưa chuẩn 0-255) ---
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# --- Morphological Closing để nối đường ---
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # kernel lớn hơn sẽ nối mạnh hơn
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

# --- Morphological Opening để loại nhiễu nhỏ ---
opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

# --- Làm dày đường để tránh bị lẹm ---
dilated = cv2.dilate(opened, kernel, iterations=1)

# --- Hiển thị ---
cv2.imshow("Original", binary)
cv2.imshow("Restored", dilated)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- Lưu kết quả ---
# cv2.imwrite("restored.png", dilated)
