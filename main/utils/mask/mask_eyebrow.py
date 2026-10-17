import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Global Detector Instance (Lazy Loading)
_app = None

def get_insightface_app():
    global _app
    if _app is None:
        _app = FaceAnalysis(name='buffalo_sc', allowed_modules=['detection'])
        _app.prepare(ctx_id=-1, det_size=(640, 640))
    return _app


# ==========================================
# Helper Functions (Gaussian & Polygon)
# ==========================================
def create_2d_gaussian_kernel(kernel_size=9, sigma=3.0):
    k_half = kernel_size // 2
    x = np.arange(-k_half, k_half + 1)
    X_k, Y_k = np.meshgrid(x, x)
    K_2d = np.exp(-(X_k**2 + Y_k**2) / (2.0 * sigma**2))
    return K_2d / np.sum(K_2d)


def apply_2d_convolution(M_binary, K_2d):
    return cv2.filter2D(
        M_binary,
        cv2.CV_32F,
        K_2d,
        borderType=cv2.BORDER_REFLECT
    )


def create_eyebrow_polygons(c_brow_right, c_brow_left, u_vec, v_vec, d_IPD):
    poly_right = np.array([
        c_brow_right + 0.35 * d_IPD * u_vec - 0.12 * d_IPD * v_vec,
        c_brow_right - 0.35 * d_IPD * u_vec - 0.12 * d_IPD * v_vec,
        c_brow_right - 0.35 * d_IPD * u_vec + 0.12 * d_IPD * v_vec,
        c_brow_right + 0.35 * d_IPD * u_vec + 0.12 * d_IPD * v_vec
    ], dtype=np.float64)

    poly_left = np.array([
        c_brow_left - 0.35 * d_IPD * u_vec - 0.12 * d_IPD * v_vec,
        c_brow_left + 0.35 * d_IPD * u_vec - 0.12 * d_IPD * v_vec,
        c_brow_left + 0.35 * d_IPD * u_vec + 0.12 * d_IPD * v_vec,
        c_brow_left - 0.35 * d_IPD * u_vec + 0.12 * d_IPD * v_vec
    ], dtype=np.float64)

    return poly_right, poly_left


# =====================================================================
# 1. ฟังก์ชัน Validate: ตรวจสอบแค่ว่าเจอคิ้ว/ใบหน้าไหม (คืนค่า 1 หรือ 0)
# =====================================================================
def validate_brow(image_matrix: np.ndarray) -> int:
    """
    ตรวจสอบว่าเจอใบหน้าเพื่อทำคิ้วหรือไม่
    Return: 1 ถ้าเจอ, 0 ถ้าไม่เจอ
    """
    if image_matrix is None:
        return 0

    try:
        app = get_insightface_app()
        faces = app.get(image_matrix)
        if len(faces) > 0 and hasattr(faces[0], 'kps') and len(faces[0].kps) >= 2:
            return 1
        return 0
    except Exception:
        return 0


# =====================================================================
# 2. ฟังก์ชัน Masking: คำนวณสร้าง Mask และ Vector Field (V_warp)
# =====================================================================
def create_brow_mask(image_matrix: np.ndarray, kernel_size=9, sigma=3.0):
    """
    สร้าง Mask และ Vector Field (V_warp) สำหรับคิ้ว
    Return: (M_display, V_warp) หรือ (None, None)
    """
    if image_matrix is None:
        return None, None

    app = get_insightface_app()
    faces = app.get(image_matrix)

    if len(faces) == 0:
        return None, None

    H, W = image_matrix.shape[:2]

    # ดึงพิกัดดวงตา
    kps = faces[0].kps
    p_eye_right = kps[0].astype(np.float64)
    p_eye_left = kps[1].astype(np.float64)

    d_IPD = np.linalg.norm(p_eye_left - p_eye_right)
    if d_IPD <= 0:
        d_IPD = 1.0

    eye_vec = p_eye_left - p_eye_right
    norm_eye = np.linalg.norm(eye_vec)
    u_vec = eye_vec / norm_eye if norm_eye > 0 else np.array([1.0, 0.0])
    v_vec = np.array([-u_vec[1], u_vec[0]], dtype=np.float64)

    # หาจุดศูนย์กลางคิ้ว
    c_brow_right = p_eye_right - 0.22 * d_IPD * v_vec
    c_brow_left = p_eye_left - 0.22 * d_IPD * v_vec

    # สร้าง Mask คิ้ว
    poly_right, poly_left = create_eyebrow_polygons(c_brow_right, c_brow_left, u_vec, v_vec, d_IPD)

    M_binary = np.zeros((H, W), dtype=np.float32)
    cv2.fillPoly(M_binary, [np.round(poly_right).astype(np.int32)], color=1.0)
    cv2.fillPoly(M_binary, [np.round(poly_left).astype(np.int32)], color=1.0)

    K_2d = create_2d_gaussian_kernel(kernel_size=kernel_size, sigma=sigma)
    M_eyebrows = apply_2d_convolution(M_binary, K_2d)
    M_eyebrows = np.clip(M_eyebrows, 0.0, 1.0)

    M_display = (M_eyebrows * 255).astype(np.uint8)

    # คำนวณ Vector Field (V_warp) สำหรับ Warp
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

    Rx = 0.40 * d_IPD
    Ry = 0.25 * d_IPD
    ky = 0.70

    dx_r = gx - c_brow_right[0]
    dy_r = gy - c_brow_right[1]
    u_r = (dx_r / Rx) ** 2 + (dy_r / Ry) ** 2
    w_r = np.where(u_r < 1.0, (1.0 - u_r) ** 2, 0.0).astype(np.float32)

    dx_l = gx - c_brow_left[0]
    dy_l = gy - c_brow_left[1]
    u_l = (dx_l / Rx) ** 2 + (dy_l / Ry) ** 2
    w_l = np.where(u_l < 1.0, (1.0 - u_l) ** 2, 0.0).astype(np.float32)

    # เวกเตอร์แรงดึงคิ้วฐาน (สเกล 1.0 = boost 0.75)
    V_x = 0.75 * (dx_r * w_r + dx_l * w_l)
    V_y = 0.75 * ky * (dy_r * w_r + dy_l * w_l)

    V_warp = np.stack([V_x, V_y], axis=-1)

    return M_display, V_warp


# =====================================================================
# 3. ฟังก์ชัน Adjust: ใช้ V_warp กับค่า alpha ปรับแต่งภาพ
# =====================================================================
def brow_adjust(image_matrix: np.ndarray, V_warp: np.ndarray, alpha: float = 0.0):
    """
    ประมวลผลปรับแต่งคิ้วโดยคำนวณเฉพาะค่า alpha กับ V_warp ที่เตรียมไว้แล้ว
    """
    H, W = image_matrix.shape[:2]
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

    map_x = (gx - alpha * V_warp[..., 0]).astype(np.float32)
    map_y = (gy - alpha * V_warp[..., 1]).astype(np.float32)

    warped_img = cv2.remap(
        image_matrix,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    return warped_img