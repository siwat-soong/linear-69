import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # ปิด PyTorch Compiler ป้องกัน Error เรื่อง cl.exe

import cv2
import face_alignment
import numpy as np
from scipy.signal import convolve2d

# Global Detector Instance (โหลดโมเดลครั้งเดียวแบบ Lazy Loading)
_fa = None

def get_fa_detector():
    global _fa
    if _fa is None:
        _fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            flip_input=False,
            device='cpu'
        )
    return _fa


# =========================================================
# Helper Functions: Coordinate & Vector Calculation
# =========================================================
def calculate_face_midline(p_nose, p_chin):
    v_midline = p_chin - p_nose
    norm_sq = np.dot(v_midline, v_midline)
    return v_midline, norm_sq

def project_point_to_line(p, p_nose, v_midline, norm_sq):
    a = p - p_nose
    c = np.sum(a * v_midline, axis=-1, keepdims=True) / norm_sq
    p_proj = p_nose + c * v_midline
    return p_proj

def calculate_displacement_direction(p, p_proj):
    d = p_proj - p
    distance = np.linalg.norm(d, axis=-1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        u = np.where(distance == 0, 0.0, d / distance)
    return d, np.squeeze(distance, axis=-1), u

def face_midline_vector(P_nose, P_chin, p):
    v_midline, norm_sq = calculate_face_midline(P_nose, P_chin)
    p_proj = project_point_to_line(p, P_nose, v_midline, norm_sq)
    d, distance, u = calculate_displacement_direction(p, p_proj)
    return d, distance, u


# =========================================================
# Soft Mask & Displacement Field Generation
# =========================================================
def create_2d_gaussian_kernel(kernel_size=15, sigma=5.0):
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(- (xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

def generate_soft_mask(image_shape, cheek_landmarks, kernel_size=15, sigma=5.0):
    binary_mask = np.zeros(image_shape, dtype=np.float32)

    for pt in cheek_landmarks:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= y < image_shape[0] and 0 <= x < image_shape[1]: 
            binary_mask[y, x] = 1.0

    kernel = create_2d_gaussian_kernel(kernel_size, sigma)
    soft_mask = convolve2d(binary_mask, kernel, mode='same', boundary='symm')
    
    max_val = np.max(soft_mask)
    if max_val > 0:
        soft_mask = soft_mask / max_val

    return soft_mask

def generate_displacement_field(image_shape, p_nose, p_chin, soft_mask_W, k=10.0):
    h, w = image_shape

    x_coord = np.arange(w, dtype=np.float32)
    y_coord = np.arange(h, dtype=np.float32)
    X, Y = np.meshgrid(x_coord, y_coord)
    P_grid = np.stack([X, Y], axis=-1)

    _, _, U_grid = face_midline_vector(p_nose, p_chin, P_grid)

    W_expanded = np.expand_dims(soft_mask_W, axis=-1)
    displacement_field = k * W_expanded * U_grid

    return displacement_field

def bilinear_interpolate(image, x_indices, y_indices):
    h, w = image.shape[:2]

    x_clip = np.clip(x_indices, 0, w - 1 - 1e-4)
    y_clip = np.clip(y_indices, 0, h - 1 - 1e-4)

    x1 = np.floor(x_clip).astype(np.int32)
    y1 = np.floor(y_clip).astype(np.int32)
    x2 = np.minimum(x1 + 1, w - 1)
    y2 = np.minimum(y1 + 1, h - 1)

    dx = x_clip - x1
    dy = y_clip - y1

    if image.ndim == 3: 
        dx = np.expand_dims(dx, axis=-1)
        dy = np.expand_dims(dy, axis=-1)

    I_11 = image[y1, x1]
    I_21 = image[y1, x2]
    I_12 = image[y2, x1]
    I_22 = image[y2, x2]

    I_out = (1 - dx) * (1 - dy) * I_11 + \
            dx * (1 - dy) * I_21 + \
            (1 - dx) * dy * I_12 + \
            dx * dy * I_22
            
    return I_out


# =====================================================================
# 1. ฟังก์ชัน Validate: ตรวจสอบแค่ว่าเจอแก้ม/ใบหน้าไหม (คืนค่า 1 หรือ 0)
# =====================================================================
def validate_cheek(image_matrix: np.ndarray) -> int:
    """
    ตรวจสอบว่าเจอจุด Landmark ของใบหน้าและแก้มหรือไม่
    Return: 1 ถ้าเจอ, 0 ถ้าไม่เจอ
    """
    if image_matrix is None:
        return 0

    try:
        fa = get_fa_detector()
        preds = fa.get_landmarks(image_matrix)

        if preds is None or len(preds) == 0:
            return 0

        landmarks = preds[0]
        if len(landmarks) > 14:
            return 1
        return 0
    except Exception:
        return 0


# =====================================================================
# 2. ฟังก์ชัน Masking: คำนวณสร้าง Mask และ Vector Field (V_warp)
# =====================================================================
def create_cheek_mask(image_matrix: np.ndarray, k_strength=10.0):
    """
    สร้าง Mask และ Vector Field (V_warp) สำหรับแก้ม
    Return: (M_cheek_display, V_warp) หรือ (None, None)
    """
    if image_matrix is None:
        return None, None

    fa = get_fa_detector()
    preds = fa.get_landmarks(image_matrix)

    if preds is None or len(preds) == 0:
        return None, None

    landmarks = preds[0]
    get_pt = lambda idx: np.array([landmarks[idx][0], landmarks[idx][1]], dtype=np.float32)

    p_nose = get_pt(30)
    p_chin = get_pt(8)

    cheek_indices = [2, 3, 4, 12, 13, 14]
    cheek_landmarks = [get_pt(idx) for idx in cheek_indices]

    h, w = image_matrix.shape[:2]

    kernel_size = int(max(w, h) * 0.08) | 1
    sigma = kernel_size / 3.0
    soft_mask_W = generate_soft_mask((h, w), cheek_landmarks, kernel_size=kernel_size, sigma=sigma)

    # สร้าง Displacement Field (V_warp)
    V_warp = generate_displacement_field((h, w), p_nose, p_chin, soft_mask_W, k=k_strength)
    M_cheek_display = (np.clip(soft_mask_W, 0.0, 1.0) * 255).astype(np.uint8)

    return M_cheek_display, V_warp


# =====================================================================
# 3. ฟังก์ชัน Adjust: ใช้ V_warp กับค่า alpha ปรับแต่งภาพ
# =====================================================================
def cheek_adjust(image_matrix: np.ndarray, V_warp: np.ndarray, alpha: float = 0.0):
    """
    ประมวลผลปรับแต่งแก้มโดยคำนวณเฉพาะค่า alpha กับ V_warp ที่เตรียมไว้แล้ว
    """
    h, w = image_matrix.shape[:2]

    x_coords = np.arange(w, dtype=np.float32)
    y_coords = np.arange(h, dtype=np.float32)
    X, Y = np.meshgrid(x_coords, y_coords)

    # คำนวณพิกัดย้อนกลับด้วย alpha scaling
    X_src = X - (alpha * V_warp[:, :, 0])
    Y_src = Y - (alpha * V_warp[:, :, 1])

    warped_img = bilinear_interpolate(image_matrix.astype(np.float32), X_src, Y_src)
    warped_img = np.clip(warped_img, 0, 255).astype(np.uint8)

    return warped_img