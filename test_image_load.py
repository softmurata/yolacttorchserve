import cv2
from PIL import Image
import numpy as np

def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

image_path = "kitten_small.jpg"

cv_image = cv2.imread(image_path)
print("opencv image shape:", cv_image.shape)

pil_image = Image.open(image_path)
pil_image = np.asarray(pil_image)
print("pil image shape:", pil_image.shape)
