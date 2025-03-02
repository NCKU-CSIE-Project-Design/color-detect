import cv2
import dlib
import numpy as np
from sklearn.cluster import KMeans

def get_dominant_color(img, mask=None):
    """使用K-means聚类方法获取主要颜色"""
    # 如果提供了遮罩，应用遮罩
    if mask is not None:
        img = cv2.bitwise_and(img, img, mask=mask)
    
    # 将图像重塑为二维数组
    pixels = img.reshape(-1, 3)
    
    # 移除黑色像素（如果有遮罩）
    if mask is not None:
        pixels = pixels[pixels.sum(axis=1) > 0]
    
    if len(pixels) == 0:
        return np.array([0, 0, 0])
    
    # 使用K-means聚类找到主要颜色
    kmeans = KMeans(n_clusters=1, n_init=10)
    kmeans.fit(pixels)
    
    # 返回主要颜色
    dominant_color = kmeans.cluster_centers_[0]
    return np.uint8(dominant_color)

def get_hair_color(img, face):
    # 計算頭髮區域（臉部上方）
    face_top = face.top()
    face_width = face.right() - face.left()
    hair_height = face_width // 2
    
    # 確保不超出圖片範圍
    hair_top = max(0, face_top - hair_height)
    hair_bottom = face_top
    hair_left = face.left()
    hair_right = face.right()
    
    # 提取頭髮區域
    hair_region = img[hair_top:hair_bottom, hair_left:hair_right]
    
    # 轉換到HSV色彩空間
    hsv_hair = cv2.cvtColor(hair_region, cv2.COLOR_RGB2HSV)
    
    # 設定頭髮顏色範圍
    lower_hair = np.array([0, 0, 0])
    upper_hair = np.array([180, 255, 100])
    
    # 創建遮罩
    mask = cv2.inRange(hsv_hair, lower_hair, upper_hair)
    
    # 獲取主要顏色
    dominant_color = get_dominant_color(hair_region, mask)
    return rgb_to_hex(dominant_color)

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
    
    # 轉換到HSV色彩空間
    hsv_face = cv2.cvtColor(face_region, cv2.COLOR_RGB2HSV)
    
    # 設定膚色範圍
    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])
    
    # 創建遮罩
    mask = cv2.inRange(hsv_face, lower_skin, upper_skin)
    
    # 獲取主要顏色
    dominant_color = get_dominant_color(face_region, mask)
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
    
    # 轉換到HSV色彩空間
    hsv_lip = cv2.cvtColor(lip_region, cv2.COLOR_RGB2HSV)
    
    # 設定嘴唇顏色範圍
    lower_lip = np.array([0, 50, 50])
    upper_lip = np.array([10, 255, 255])
    
    # 創建遮罩
    mask = cv2.inRange(hsv_lip, lower_lip, upper_lip)
    
    # 獲取主要顏色
    dominant_color = get_dominant_color(lip_region, mask)
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
