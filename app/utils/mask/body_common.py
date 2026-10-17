import cv2
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POSE_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'pose_landmarker_full.task')

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24

VISIBILITY_THRESHOLD = 0.65
POLY_FEATHER = 25  # px, เกลี่ยขอบ polygon ลำตัว

# Global Pose Landmarker Instance (Lazy Loading)
_landmarker = None

def get_pose_landmarker():
    global _landmarker
    if _landmarker is None:
        base_options = BaseOptions(model_asset_path=POSE_MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
        )
        _landmarker = vision.PoseLandmarker.create_from_options(options)
    return _landmarker


# ============================================================
# Math Helpers & Grid Calculation
# ============================================================
def pixel_grid(h, w):
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return np.stack([X, Y], axis=-1)

def project_on_axis(P, A, B):
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    AB = B - A
    L2 = float(AB @ AB)

    t_raw = ((P - A) @ AB) / max(L2, 1e-6)
    t_clip = np.clip(t_raw, 0.0, 1.0)
    Q = A + t_clip[..., None] * AB
    return Q.astype(np.float32), t_raw.astype(np.float32)

def gaussian_band(t, t_center, sigma):
    return np.exp(-((t - t_center) ** 2) / (2.0 * sigma ** 2))

def point_in_polygon_mask(h, w, poly_pts):
    P = pixel_grid(h, w)
    pts = np.asarray(poly_pts, dtype=np.float32)
    n = len(pts)

    sign_sum = np.zeros((h, w), dtype=np.float32)
    for i in range(n):
        V0 = pts[i]
        V1 = pts[(i + 1) % n]
        edge = V1 - V0
        to_pixel = P - V0
        cross = edge[0] * to_pixel[..., 1] - edge[1] * to_pixel[..., 0]
        sign_sum += np.sign(cross)

    M_poly = (np.abs(sign_sum) >= n - 0.5).astype(np.float32)
    return M_poly

def build_torso_polygon_mask(h, w, L_sh, R_sh, R_hip, L_hip):
    M_hard = point_in_polygon_mask(h, w, [L_sh, R_sh, R_hip, L_hip])
    k = 2 * POLY_FEATHER + 1
    M_soft = cv2.dilate(M_hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    M_soft = np.clip(cv2.GaussianBlur(M_soft, (0, 0), POLY_FEATHER / 3), 0, 1)
    return M_soft

def bilinear_sample(img, map_x, map_y):
    h, w = img.shape[:2]
    x = np.clip(map_x, 0, w - 1.001)
    y = np.clip(map_y, 0, h - 1.001)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    wx = (x - x0)[..., None]
    wy = (y - y0)[..., None]

    f = img.astype(np.float32)
    top = f[y0, x0] * (1 - wx) + f[y0, x0 + 1] * wx
    bot = f[y0 + 1, x0] * (1 - wx) + f[y0 + 1, x0 + 1] * wx

    return np.clip(top * (1 - wy) + bot * wy, 0, 255).astype(np.uint8)


# ============================================================
# 1. Validate Body (สำหรับ Chest, Waist, Hips)
# ============================================================
def validate_body(image_matrix: np.ndarray) -> int:
    """
    ตรวจสอบความเห็นชัดของไหล่และสะโพกในภาพ
    Return: 1 ถ้าเห็นชัดเจน, 0 ถ้าไม่เห็น
    """
    if image_matrix is None:
        return 0

    try:
        h, w = image_matrix.shape[:2]
        img_rgb = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2RGB)
        img_rgb = np.ascontiguousarray(img_rgb)  # ป้องกันปัญหา Memory Layout

        landmarker = get_pose_landmarker()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = landmarker.detect(mp_image)

        if not result.pose_landmarks:
            print("[DEBUG validate_body] MediaPipe ตรวจไม่พบโครงสร้างร่างกายในภาพ")
            return 0

        lm = result.pose_landmarks[0]
        vis_needed = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
        min_vis = min(lm[i].visibility for i in vis_needed)

        if min_vis < VISIBILITY_THRESHOLD:
            print(f"[DEBUG validate_body] ค่า Visibility ต่ำไป ({min_vis:.2f} < {VISIBILITY_THRESHOLD})")
            return 0

        print(f"[DEBUG validate_body] Validation ผ่านสำเร็จ (Visibility Min: {min_vis:.2f})")
        return 1

    except Exception as e:
        # พิมพ์ Error ออกมาทาง Terminal เพื่อให้ตรวจสอบสาเหตุที่แท้จริงได้
        print(f"[ERROR validate_body] เกิดข้อผิดพลาด: {e}")
        return 0

# ============================================================
# 2. Create Body Mask & Cache Data
# ============================================================
def create_body_mask(image_matrix: np.ndarray):
    """
    สร้าง Mask และดึงแกนโปรเจกต์ของลำตัวเตรียมเก็บเข้า Cache
    Return: (M_display, body_data_dict)
    """
    if image_matrix is None:
        return None, None

    h, w = image_matrix.shape[:2]
    img_rgb = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2RGB)

    landmarker = get_pose_landmarker()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        return None, None

    lm = result.pose_landmarks[0]
    vis_needed = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
    if min(lm[i].visibility for i in vis_needed) < VISIBILITY_THRESHOLD:
        return None, None

    def pt(idx):
        return np.array([lm[idx].x * w, lm[idx].y * h], dtype=np.float32)

    L_sh, R_sh = pt(LEFT_SHOULDER), pt(RIGHT_SHOULDER)
    L_hip, R_hip = pt(LEFT_HIP), pt(RIGHT_HIP)

    Sc = (L_sh + R_sh) / 2.0
    Hc = (L_hip + R_hip) / 2.0

    seg_mask = result.segmentation_masks[0].numpy_view()
    M_person = (seg_mask >= 0.30).astype(np.float32)
    k = 2 * POLY_FEATHER + 1
    M_person = cv2.dilate(M_person, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    M_person = np.clip(cv2.GaussianBlur(M_person, (0, 0), POLY_FEATHER / 3), 0, 1)

    M_torso = build_torso_polygon_mask(h, w, L_sh, R_sh, R_hip, L_hip)
    G_body = M_torso * M_person

    P = pixel_grid(h, w)
    Q, t_raw = project_on_axis(P, Sc, Hc)

    M_display = (np.clip(G_body, 0.0, 1.0) * 255).astype(np.uint8)

    body_cache_data = {
        'P': P, 'Q': Q, 't_raw': t_raw, 'G_body': G_body
    }

    return M_display, body_cache_data


# ============================================================
# 3. Adjust Single Body Part
# ============================================================
def adjust_body_part(image_matrix: np.ndarray, body_cache_data: dict, scale_val: float, t_center: float, sigma: float = 0.15):
    """
    ปรับแต่งขยาย/บีบลำตัวเฉพาะส่วน (Chest: 0.25, Waist: 0.55, Hips: 0.85)
    """
    scale = 1.0 - (scale_val / 100.0)
    if abs(scale - 1.0) < 1e-4:
        return image_matrix.copy()

    P = body_cache_data['P']
    Q = body_cache_data['Q']
    t_raw = body_cache_data['t_raw']
    G_body = body_cache_data['G_body']

    G_band = gaussian_band(t_raw, t_center, sigma) * G_body
    s_field = 1.0 + (scale - 1.0) * G_band
    s_field = np.maximum(s_field, 0.05)

    perp = P - Q
    source = Q + perp / s_field[..., None]

    return bilinear_sample(image_matrix, source[..., 0], source[..., 1])