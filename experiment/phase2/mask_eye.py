import sys
import os
import warnings

# ปิด Warning และ Exception ignored
sys.unraisablehook = lambda unraisable: None
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import cv2
import numpy as np
from mtcnn import MTCNN

detector = MTCNN()
img_path = './dataset/img_0002_jpeg.jpg'
img = cv2.imread(img_path)

if img is None: 
    print('Image Not Found')
    sys.exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
height, width = img.shape[:2]

result = detector.detect_faces(img_rgb)

if len(result) > 0:
    keypoints = result[0]['keypoints']
    x_R, y_R = int(keypoints['right_eye'][0]), int(keypoints['right_eye'][1])
    x_L, y_L = int(keypoints['left_eye'][0]), int(keypoints['left_eye'][1])