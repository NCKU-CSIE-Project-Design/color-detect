import cv2
import dlib
import numpy as np
from sklearn.cluster import KMeans
import cv2
import dlib

# 讀取圖像
img = cv2.imread('input_image.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 初始化人臉檢測器
detector = dlib.get_frontal_face_detector()

# 檢測人臉
faces = detector(img_rgb)

# 顯示檢測到的臉部框
for face in faces:
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 縮小圖像（50%）
small_img = cv2.resize(img, (int(img.shape[1] * 0.25), int(img.shape[0] * 0.25)))

# 顯示縮小後的圖像
cv2.imshow("Detected Faces", small_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


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
    
    # 獲取各個部位的顏色
    colors = {
        '膚色': get_skin_color(img, face),
        '眼睛': get_eye_color(img, landmarks),
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


def get_eye_color(img, landmarks):
    # 提取眼睛區域的點
    left_eye = []
    for i in range(36, 42):
        left_eye.append((landmarks.part(i).x, landmarks.part(i).y))
    
    # 計算眼睛區域的平均顏色
    eye_color = np.mean([img[y, x] for x, y in left_eye], axis=0)
    return rgb_to_hex(eye_color)

def get_lip_color(img, landmarks):
    # 提取嘴唇區域的點
    lips = []
    for i in range(48, 60):
        lips.append((landmarks.part(i).x, landmarks.part(i).y))
    
    # 計算嘴唇區域的平均顏色
    lip_color = np.mean([img[y, x] for x, y in lips], axis=0)
    return rgb_to_hex(lip_color)

def rgb_to_hex(rgb):
    # 將 RGB 值轉換為十六進制色碼
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

# 使用示例
image_path = 'input_image.jpg'
colors = get_facial_colors(image_path)
print('臉部顏色分析結果：')
for part, color in colors.items():
    print(f'{part}: {color}')