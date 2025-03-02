import cv2
import dlib
import numpy as np
from sklearn.cluster import KMeans
import cv2
import dlib


def visualize_eye_region(img, landmarks, save_path='eye_region.jpg'):
    # 複製原圖，避免修改原始圖片
    img_copy = img.copy()
    
    # 提取眼睛區域的點
    left_eye = []
    for i in range(36, 42):
        left_eye.append((landmarks.part(i).x, landmarks.part(i).y))
    
    # 計算眼睛區域的中心點
    x_coords = [x for x, y in left_eye]
    y_coords = [y for x, y in left_eye]
    eye_center_x = sum(x_coords) // len(x_coords)
    eye_center_y = sum(y_coords) // len(y_coords)
    
    # 計算瞳孔檢測區域
    radius = (max(x_coords) - min(x_coords)) // 4
    x1 = max(0, eye_center_x - radius)
    x2 = min(img.shape[1], eye_center_x + radius)
    y1 = max(0, eye_center_y - radius)
    y2 = min(img.shape[0], eye_center_y + radius)
    
    # 畫出眼睛輪廓點
    for x, y in left_eye:
        cv2.circle(img_copy, (x, y), 2, (0, 255, 0), -1)
    
    # 畫出眼睛中心點
    cv2.circle(img_copy, (eye_center_x, eye_center_y), 2, (255, 0, 0), -1)
    
    # 畫出瞳孔檢測區域
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 1)
    
    # 將圖片從RGB轉換為BGR以正確保存顏色
    img_bgr = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
    
    # 保存圖片
    cv2.imwrite(save_path, img_bgr)

def get_hair_color(img, face):
    # 計算頭髮區域（臉部上方）
    face_top = face.top()
    face_width = face.right() - face.left()
    hair_height = face_width // 2  # 設定頭髮區域高度
    
    # 確保不超出圖片範圍
    hair_top = max(0, face_top - hair_height)
    hair_bottom = face_top
    hair_left = face.left()
    hair_right = face.right()
    
    # 提取頭髮區域
    hair_region = img[hair_top:hair_bottom, hair_left:hair_right]
    
    # 轉換到HSV色彩空間
    hsv_hair = cv2.cvtColor(hair_region, cv2.COLOR_RGB2HSV)
    
    # 設定頭髮顏色範圍（較暗的顏色）
    lower_hair = np.array([0, 0, 0])
    upper_hair = np.array([180, 255, 100])  # 調整亮度範圍以捕捉頭髮顏色
    
    # 創建遮罩
    mask = cv2.inRange(hsv_hair, lower_hair, upper_hair)
    
    # 提取頭髮區域
    hair = cv2.bitwise_and(hair_region, hair_region, mask=mask)
    
    # 移除黑色像素
    valid_pixels = hair[np.sum(hair, axis=2) > 30]
    
    if len(valid_pixels) == 0:
        return '#000000'
    
    # 找出最常見的顏色
    unique_colors, counts = np.unique(valid_pixels, axis=0, return_counts=True)
    most_common_color = unique_colors[np.argmax(counts)]
    
    return rgb_to_hex(most_common_color)

def visualize_facial_regions(img, face, landmarks, save_path='facial_regions.jpg'):
    # 複製原圖，避免修改原始圖片
    img_copy = img.copy()
    
    # 繪製臉部框
    cv2.rectangle(img_copy, 
                 (face.left(), face.top()),
                 (face.right(), face.bottom()),
                 (0, 255, 0), 2)
    
    # 繪製頭髮區域
    face_width = face.right() - face.left()
    hair_height = face_width // 2
    hair_top = max(0, face.top() - hair_height)
    cv2.rectangle(img_copy,
                 (face.left(), hair_top),
                 (face.right(), face.top()),
                 (255, 0, 0), 2)
    
    # 繪製嘴唇區域
    lips = []
    for i in range(48, 60):
        lips.append((landmarks.part(i).x, landmarks.part(i).y))
        cv2.circle(img_copy, (landmarks.part(i).x, landmarks.part(i).y), 2, (0, 255, 0), -1)
    
    x_coords = [x for x, y in lips]
    y_coords = [y for x, y in lips]
    x1, x2 = min(x_coords), max(x_coords)
    y1, y2 = min(y_coords), max(y_coords)
    padding = 5
    
    cv2.rectangle(img_copy,
                 (x1 - padding, y1 - padding),
                 (x2 + padding, y2 + padding),
                 (255, 255, 0), 2)
    
    # 添加標籤
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_copy, 'Hair', (face.left(), hair_top - 10), font, 0.5, (255, 0, 0), 2)
    cv2.putText(img_copy, 'Face', (face.left(), face.top() - 10), font, 0.5, (0, 255, 0), 2)
    cv2.putText(img_copy, 'Lips', (x1 - padding, y1 - padding - 10), font, 0.5, (255, 255, 0), 2)
    
    # 將圖片從RGB轉換為BGR以正確保存顏色
    img_bgr = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
    
    # 保存圖片
    cv2.imwrite(save_path, img_bgr)

def get_facial_colors(image_path):
    # 讀取圖片
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 初始化人臉檢測器和特徵點檢測器
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    
    # 檢測人臉
    faces = detector(img)
    if len(faces) == 0:
        return None
    
    face = faces[0]
    landmarks = predictor(img, face)
    
    # 可視化所有區域並保存
    visualize_facial_regions(img, face, landmarks, 'facial_regions.jpg')
    
    # 獲取各個部位的顏色
    colors = {
        '頭髮': get_hair_color(img, face),
        '膚色': get_skin_color(img, face),
        '嘴唇': get_lip_color(img, landmarks)
    }
    
    return colors

def get_skin_color(img, face):
    # 提取臉部區域
    face_region = img[face.top():face.bottom(), face.left():face.right()]

    # 將圖像從 RGB 轉換為 HSV 色彩空間
    hsv_face = cv2.cvtColor(face_region, cv2.COLOR_RGB2HSV)
    
    # 設定膚色範圍（此範圍適用於大多數膚色）
    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])

    # 創建遮罩
    mask = cv2.inRange(hsv_face, lower_skin, upper_skin)
    
    # 提取膚色區域
    skin = cv2.bitwise_and(face_region, face_region, mask=mask)
    
    # 使用 K-means 聚類來獲取膚色
    pixels = skin.reshape(-1, 3)
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(pixels)
    
    # 返回最主要的顏色（中心點）
    dominant_color = kmeans.cluster_centers_[0]
    return rgb_to_hex(dominant_color)

def get_lip_color(img, landmarks):
    # 提取嘴唇區域的點
    lips = []
    for i in range(48, 60):
        lips.append((landmarks.part(i).x, landmarks.part(i).y))
    
    # 計算嘴唇區域的邊界
    x_coords = [x for x, y in lips]
    y_coords = [y for x, y in lips]
    x1, x2 = min(x_coords), max(x_coords)
    y1, y2 = min(y_coords), max(y_coords)
    
    # 擴大區域以確保完全包含嘴唇
    padding = 5
    lip_region = img[max(0, y1-padding):min(img.shape[0], y2+padding), 
                    max(0, x1-padding):min(img.shape[1], x2+padding)]
    
    # 將圖像轉換為HSV色彩空間
    hsv_lip = cv2.cvtColor(lip_region, cv2.COLOR_RGB2HSV)
    
    # 設定嘴唇顏色範圍（可以根據需要調整）
    lower_lip = np.array([0, 50, 50])
    upper_lip = np.array([10, 255, 255])
    
    # 創建遮罩
    mask = cv2.inRange(hsv_lip, lower_lip, upper_lip)
    
    # 提取嘴唇區域
    lip = cv2.bitwise_and(lip_region, lip_region, mask=mask)
    
    # 使用K-means聚類
    pixels = lip.reshape(-1, 3)
    pixels = pixels[pixels.sum(axis=1) > 0]  # 移除黑色像素
    if len(pixels) == 0:
        return '#000000'  # 如果沒有有效像素，返回黑色
    
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(pixels)
    
    # 返回最主要的顏色
    dominant_color = kmeans.cluster_centers_[0]
    return rgb_to_hex(dominant_color)

def rgb_to_hex(rgb):
    # 將 RGB 值轉換為十六進制色碼
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

# 使用示例
image_path = 'input_image.png'
colors = get_facial_colors(image_path)
print('臉部顏色分析結果：')
for part, color in colors.items():
    print(f'{part}: {color}')