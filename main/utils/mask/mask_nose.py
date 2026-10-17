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
def create_2d_gaussian_kernel(kernel_size=3, sigma=1.0):
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


def create_nose_polygon(
    p_nose_bridge,
    p_nostril_right,
    p_nostril_left,
    p_eye_right,
    p_eye_left,
    d_IPD
):
    p_nose_bridge = np.array(p_nose_bridge, dtype=np.float64)
    p_nostril_right = np.array(p_nostril_right, dtype=np.float64)
    p_nostril_left = np.array(p_nostril_left, dtype=np.float64)
    p_eye_right = np.array(p_eye_right, dtype=np.float64)
    p_eye_left = np.array(p_eye_left, dtype=np.float64)

    eye_vec = p_eye_left - p_eye_right
    norm_eye = np.linalg.norm(eye_vec)

    if norm_eye == 0:
        u_vec = np.array([1.0, 0.0], dtype=np.float64)
    else:
        u_vec = eye_vec / norm_eye

    v_vec = np.array([-u_vec[1], u_vec[0]], dtype=np.float64)

    N1 = p_nose_bridge - 0.15 * d_IPD * u_vec
    N2 = p_nose_bridge + 0.15 * d_IPD * u_vec
    N3 = p_nostril_right + 0.10 * d_IPD * u_vec + 0.05 * d_IPD * v_vec
    N4 = p_nostril_left - 0.10 * d_IPD * u_vec + 0.05 * d_IPD * v_vec

    return np.array([N1, N2, N3, N4], dtype=np.float64)


# =====================================================================
# 1. ฟังก์ชัน Validate: ตรวจสอบแค่ว่าเจอจมูก/ใบหน้าไหม (คืนค่า 1 หรือ 0)
# =====================================================================
def validate_nose(image_matrix: np.ndarray) -> int:
    """
    ตรวจสอบว่าเจอใบหน้าและจุดตำแหน่งจมูกหรือไม่
    Return: 1 ถ้าเจอ, 0 ถ้าไม่เจอ
    """
    if image_matrix is None:
        return 0

    try:
        app = get_insightface_app()
        faces = app.get(image_matrix)
        if len(faces) > 0 and hasattr(faces[0], 'kps') and len(faces[0].kps) >= 3:
            return 1
        return 0
    except Exception:
        return 0


# =====================================================================
# 2. ฟังก์ชัน Masking: คำนวณสร้าง Mask และ Vector Field (V_warp)
# =====================================================================
def create_nose_mask(image_matrix: np.ndarray, kernel_size=3, sigma=1.0, ky=0.35, rx_k=0.9, ry_k=0.5, cy_pos=0.65):
    """
    สร้าง Mask และ Vector Field (V_warp) สำหรับจมูก
    Return: (M_display, V_warp) หรือ (None, None)
    """
    if image_matrix is None:
        return None, None

    app = get_insightface_app()
    faces = app.get(image_matrix)

    if len(faces) == 0:
        return None, None

    H, W = image_matrix.shape[:2]

    kps = faces[0].kps
    p_eye_right = kps[0].astype(np.float64)
    p_eye_left = kps[1].astype(np.float64)
    p_nose_tip = kps[2].astype(np.float64)

    d_IPD = np.linalg.norm(p_eye_left - p_eye_right)
    if d_IPD <= 0:
        return None, None

    p_eye_mid = (p_eye_right + p_eye_left) / 2.0
    p_nose_bridge = p_eye_mid + 0.35 * (p_nose_tip - p_eye_mid)

    eye_vec = p_eye_left - p_eye_right
    norm_eye = np.linalg.norm(eye_vec)
    u_vec = eye_vec / norm_eye if norm_eye > 0 else np.array([1.0, 0.0])

    p_nostril_right = p_nose_tip - 0.15 * d_IPD * u_vec
    p_nostril_left = p_nose_tip + 0.15 * d_IPD * u_vec

    polygon_points = create_nose_polygon(
        p_nose_bridge, p_nostril_right, p_nostril_left,
        p_eye_right, p_eye_left, d_IPD
    )

    polygon = np.round(polygon_points).astype(np.int32)
    M_binary = np.zeros((H, W), dtype=np.float32)
    cv2.fillPoly(M_binary, [polygon], color=1.0)

    K_2d = create_2d_gaussian_kernel(kernel_size=kernel_size, sigma=sigma)
    M_nose = apply_2d_convolution(M_binary, K_2d)
    M_nose = np.clip(M_nose, 0.0, 1.0)

    M_display = (M_nose * 255).astype(np.uint8)

    # คำนวณ Vector Displacement Field (V_warp)
    ys, xs = np.where(M_nose > 0.1)
    if len(xs) == 0:
        return None, None

    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    nose_w, nose_h = x1 - x0 + 1, y1 - y0 + 1

    cx = (x0 + x1) / 2.0
    cy = y0 + cy_pos * nose_h
    Rx, Ry = rx_k * nose_w, ry_k * nose_h

    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    dx, dy = gx - cx, gy - cy

    u = (dx / Rx) ** 2 + (dy / Ry) ** 2
    w = np.where(u < 1.0, (1.0 - u) ** 2, 0.0).astype(np.float32)

    V_x = dx * w
    V_y = dy * ky * w

    V_warp = np.stack([V_x, V_y], axis=-1)

    return M_display, V_warp


# =====================================================================
# 3. ฟังก์ชัน Adjust: ใช้ V_warp กับค่า alpha ปรับแต่งภาพ
# =====================================================================
def nose_adjust(image_matrix: np.ndarray, V_warp: np.ndarray, alpha: float = 0.0):
    """
    ประมวลผลปรับแต่งทรงจมูกโดยคำนวณเฉพาะค่า alpha กับ V_warp ที่เตรียมไว้แล้ว
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