import cv2
import numpy as np
from mtcnn import MTCNN

# Global Detector Instance (โหลดโมเดลครั้งเดียว)
_detector = None

def get_detector():
    global _detector
    if _detector is None:
        _detector = MTCNN()
    return _detector

def create_2d_gaussian_kernel(kernel_size=21, sigma=5.0):
    k_half = kernel_size // 2
    x = np.arange(-k_half, k_half + 1)
    X_k, Y_k = np.meshgrid(x, x)
    K_2d = np.exp(-(X_k**2 + Y_k**2) / (2.0 * (sigma**2)))
    return K_2d / np.sum(K_2d)

def apply_2d_convolution(M_binary, K_2d):
    return cv2.filter2D(M_binary, -1, K_2d, borderType=cv2.BORDER_REFLECT)

def bilinear_interpolate_vectorized(img, P_src):
    H, W = img.shape[:2]
    X_src, Y_src = P_src[..., 0], P_src[..., 1]
    
    x0 = np.clip(np.floor(X_src).astype(np.int32), 0, W - 1)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y0 = np.clip(np.floor(Y_src).astype(np.int32), 0, H - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    
    wx = (X_src - x0)[..., np.newaxis]
    wy = (Y_src - y0)[..., np.newaxis]
    
    top = (1.0 - wx) * img[y0, x0] + wx * img[y0, x1]
    bottom = (1.0 - wx) * img[y1, x0] + wx * img[y1, x1]
    
    return np.clip((1.0 - wy) * top + wy * bottom, 0, 255).astype(np.uint8)


# =====================================================================
# 1. ฟังก์ชัน Validate: ตรวจสอบแค่ว่ามีตาในภาพไหม (คืนค่า 1 หรือ 0)
# =====================================================================
def validate_eye(image_matrix: np.ndarray) -> int:
    """
    ตรวจสอบว่าเจอดวงตาในภาพหรือไม่ (ไม่ทำ Masking)
    Return: 1 ถ้าเจอดวงตา, 0 ถ้าไม่เจอดวงตา
    """
    if image_matrix is None:
        return 0

    if len(image_matrix.shape) == 3 and image_matrix.shape[2] == 3:
        img_rgb = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = image_matrix

    detector = get_detector()
    results = detector.detect_faces(img_rgb)

    if not results:
        return 0

    keypoints = results[0]['keypoints']
    if 'right_eye' in keypoints and 'left_eye' in keypoints:
        return 1
    
    return 0


# =====================================================================
# 2. ฟังก์ชัน Masking: คำนวณสร้าง Mask และ Vector Field (V_warp)
# =====================================================================
def create_eye_mask(image_matrix: np.ndarray, scale_val=0, kernel_size=25, sigma_conv=5.0):
    """
    แยกการทำ Mask และสร้าง V_warp ออกมาจาก validate
    Return: (M_eye_display, V_warp) หรือ (None, None)
    """
    if len(image_matrix.shape) == 3 and image_matrix.shape[2] == 3:
        img_rgb = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = image_matrix

    detector = get_detector()
    results = detector.detect_faces(img_rgb)

    if not results:
        return None, None

    keypoints = results[0]['keypoints']
    if 'right_eye' not in keypoints or 'left_eye' not in keypoints:
        return None, None

    p_R = np.array(keypoints['right_eye'], dtype=np.float64)
    p_L = np.array(keypoints['left_eye'], dtype=np.float64)

    H, W = image_matrix.shape[:2]

    # 1. Coordinate Grid & Displacement Fields
    X_grid, Y_grid = np.meshgrid(np.arange(W), np.arange(H))
    P_grid = np.stack([X_grid, Y_grid], axis=-1).astype(np.float64)
    v_R, v_L = P_grid - p_R, P_grid - p_L

    # 2. Rotation Matrix
    diff = p_L - p_R
    theta = np.arctan2(diff[1], diff[0])
    R_theta = np.array([
        [ np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ], dtype=np.float64)

    v_prime_R = v_R @ R_theta.T
    v_prime_L = v_L @ R_theta.T

    # 3. Distance Matrix & Gaussian Mask
    d_IPD = np.linalg.norm(diff, ord=2)
    a_e, b_e = 0.35 * d_IPD, 0.22 * d_IPD
    k = 1.0 + (scale_val / 100.0)
    
    A_scaled = (1.0 / (k**2)) * np.diag([1.0 / (a_e**2), 1.0 / (b_e**2)])

    E_R = np.sum((v_prime_R @ A_scaled) * v_prime_R, axis=-1)
    E_L = np.sum((v_prime_L @ A_scaled) * v_prime_L, axis=-1)

    M_hard_R = np.where(E_R <= 1.0, 1.0, 0.0)
    M_hard_L = np.where(E_L <= 1.0, 1.0, 0.0)

    K_2d = create_2d_gaussian_kernel(kernel_size=kernel_size, sigma=sigma_conv)

    M_R = apply_2d_convolution(M_hard_R, K_2d)[..., np.newaxis]
    M_L = apply_2d_convolution(M_hard_L, K_2d)[..., np.newaxis]

    V_warp = (M_R * v_R) + (M_L * v_L)
    M_eye_display = (np.clip(np.maximum(M_R[..., 0], M_L[..., 0]), 0.0, 1.0) * 255).astype(np.uint8)

    return M_eye_display, V_warp


# =====================================================================
# 3. ฟังก์ชัน Adjust: ใช้ V_warp กับค่า alpha ปรับแต่งภาพ
# =====================================================================
def eye_adjust(image_matrix: np.ndarray, V_warp: np.ndarray, alpha: float = 0.0):
    """
    ประมวลผลปรับแต่งรูปทรงดวงตาโดยคำนวณเฉพาะค่า alpha กับ V_warp ที่เตรียมไว้แล้ว
    """
    H, W, _ = image_matrix.shape
    X_grid, Y_grid = np.meshgrid(np.arange(W), np.arange(H))
    P_grid = np.stack([X_grid, Y_grid], axis=-1).astype(np.float64)

    P_src = P_grid - (alpha * V_warp)
    warped_img = bilinear_interpolate_vectorized(image_matrix, P_src)

    return warped_img