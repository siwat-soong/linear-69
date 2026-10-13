import sys
import os
import warnings

# ปิด Warning ของ Console
sys.unraisablehook = lambda unraisable: None
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import cv2
import numpy as np
from mtcnn import MTCNN

detector = MTCNN()
img = cv2.imread('./dataset/img_0002_jpeg.jpg')

if img is None:
    print("ไม่พบไฟล์รูปภาพ")
    sys.exit()

height, width = img.shape[:2]
results = detector.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

if len(results) > 0:
    keypoints = results[0]['keypoints']
    
    p_R = np.array(keypoints['right_eye'], dtype=np.float64)
    p_L = np.array(keypoints['left_eye'], dtype=np.float64)
    
    d_IPD = np.linalg.norm(p_R - p_L, ord=2)
    a_e = 0.35 * d_IPD
    b_e = 0.22 * d_IPD

    diff_vector = p_L - p_R
    theta_face = np.arctan2(diff_vector[1], diff_vector[0])

    R_theta = np.array([
        [ np.cos(theta_face), np.sin(theta_face)],
        [-np.sin(theta_face), np.cos(theta_face)]
    ], dtype=np.float64)

    X_grid, Y_grid = np.meshgrid(np.arange(width), np.arange(height))
    P_grid = np.stack([X_grid, Y_grid], axis=-1)

    v_R = P_grid - p_R
    v_L = P_grid - p_L

    v_prime_R = v_R @ R_theta.T
    v_prime_L = v_L @ R_theta.T

    A = np.diag([1.0 / (a_e**2), 1.0 / (b_e**2)])
    E_right = np.sum((v_prime_R @ A) * v_prime_R, axis=-1)
    E_left  = np.sum((v_prime_L @ A) * v_prime_L, axis=-1)

    E_eye = np.minimum(E_right, E_left)

    sigma_e = 0.5
    M_eye_float = np.where(
        E_eye <= 1.0, 
        1.0, 
        np.exp(-(E_eye - 1.0) / (2.0 * (sigma_e**2)))
    )

    M_eye_display = (M_eye_float * 255).astype(np.uint8)

    # แสดงรูปภาพ Mask ผลลัพธ์
    cv2.imshow('Soft Gaussian Eye Mask (Rotated)', M_eye_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("ไม่พบใบหน้าในภาพ")