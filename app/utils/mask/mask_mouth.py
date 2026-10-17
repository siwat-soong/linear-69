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
    H, W, C = img.shape
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
# 1. ฟังก์ชัน Validate: ตรวจสอบแค่ว่าเจอปากในภาพไหม (คืนค่า 1 หรือ 0)
# =====================================================================
def validate_mouth(image_matrix: np.ndarray) -> int:
    """
    ตรวจสอบว่าเจอมุมปากในภาพหรือไม่ (ไม่ทำ Masking)
    Return: 1 ถ้าเจอปาก, 0 ถ้าไม่เจอปาก
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
    if 'mouth_right' in keypoints and 'mouth_left' in keypoints:
        return 1
    
    return 0


# =====================================================================
# 2. ฟังก์ชัน Masking: คำนวณสร้าง Mask และ Vector Field (V_warp)
# =====================================================================
def create_mouth_mask(image_matrix: np.ndarray, scale_val=0, kernel_size=25, sigma_conv=6.0):
    """
    สร้าง Mask และ Vector Field (V_warp) สำหรับริมฝีปาก
    Return: (M_mouth_display, V_warp) หรือ (None, None)
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
    if 'mouth_right' not in keypoints or 'mouth_left' not in keypoints:
        return None, None

    p_MR = np.array(keypoints['mouth_right'], dtype=np.float64)
    p_ML = np.array(keypoints['mouth_left'], dtype=np.float64)

    H, W, _ = image_matrix.shape

    # 1. Coordinate Grid & Displacement Vector เทียบกับจุดศูนย์กลางปาก
    X_grid, Y_grid = np.meshgrid(np.arange(W), np.arange(H))
    P_grid = np.stack([X_grid, Y_grid], axis=-1).astype(np.float64)

    p_M = (p_MR + p_ML) / 2.0  # จุดศูนย์กลางปาก
    v_M = P_grid - p_M         # เวกเตอร์ระยะห่างจากจุดศูนย์กลางปาก

    # 2. Rotation Matrix ตามมุมเอียงของปาก
    diff = p_ML - p_MR
    theta = np.arctan2(diff[1], diff[0])
    R_theta = np.array([
        [ np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ], dtype=np.float64)

    v_prime_M = v_M @ R_theta.T

    # 3. Quadratic Form Distance Matrix (สัดส่วนวงรีของปาก)
    d_mouth = np.linalg.norm(diff, ord=2)
    a_m, b_m = 0.60 * d_mouth, 0.30 * d_mouth
    k = 1.0 + (scale_val / 100.0)

    A_scaled = (1.0 / (k**2)) * np.diag([1.0 / (a_m**2), 1.0 / (b_m**2)])

    E_M = np.sum((v_prime_M @ A_scaled) * v_prime_M, axis=-1)

    # 4. Soft Mask Generation
    M_hard_M = np.where(E_M <= 1.0, 1.0, 0.0)
    K_2d = create_2d_gaussian_kernel(kernel_size=kernel_size, sigma=sigma_conv)
    M_M = apply_2d_convolution(M_hard_M, K_2d)[..., np.newaxis]

    # 5. Vector Displacement Field
    V_warp = M_M * v_M
    M_mouth_display = (np.clip(M_M[..., 0], 0.0, 1.0) * 255).astype(np.uint8)

    return M_mouth_display, V_warp


# =====================================================================
# 3. ฟังก์ชัน Adjust: ใช้ V_warp กับค่า alpha ปรับแต่งภาพ
# =====================================================================
def mouth_adjust(image_matrix: np.ndarray, V_warp: np.ndarray, alpha: float = 0.0):
    """
    ประมวลผลปรับแต่งริมฝีปากโดยคำนวณเฉพาะค่า alpha กับ V_warp ที่เตรียมไว้แล้ว
    """
    H, W, _ = image_matrix.shape
    X_grid, Y_grid = np.meshgrid(np.arange(W), np.arange(H))
    P_grid = np.stack([X_grid, Y_grid], axis=-1).astype(np.float64)

    P_src = P_grid - (alpha * V_warp)
    warped_img = bilinear_interpolate_vectorized(image_matrix, P_src)

    return warped_img