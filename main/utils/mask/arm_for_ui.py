import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# Dynamic Absolute Path สำหรับไฟล์โมเดล Pose
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'pose_landmarker_full.task')
POSE_MODEL_PATH = os.getenv('POSE_MODEL_PATH', DEFAULT_MODEL_PATH)

# Index ของ Landmarks สำหรับแขนและลำตัว
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24

UPPER_RADIUS = 22
LOWER_RADIUS = 18
FEATHER = 25
VISIBILITY_THRESHOLD = 0.35  # ปรับลดเพื่อให้ตรวจจับในสภาวะจริงได้ง่ายขึ้น

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


def closest_point_on_segment(P, A, B):
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    AB = B - A
    L2 = float(AB @ AB)

    if L2 < 1e-6:
        Q = np.broadcast_to(A, P.shape).copy()
    else:
        AP = P - A
        t = np.clip((AP @ AB) / L2, 0.0, 1.0)
        Q = A + t[..., None] * AB

    D = np.linalg.norm(P - Q, axis=-1)
    return Q.astype(np.float32), D.astype(np.float32)


def weight_W(r, R):
    q = np.clip(r / R, 0.0, 1.0)
    return (1.0 - q ** 2) ** 2


def build_arm_field(h, w, S, E, Wr, upper_ok, lower_ok):
    P = pixel_grid(h, w)

    Qu = Du = Ql = Dl = None
    if upper_ok:
        Qu, Du = closest_point_on_segment(P, S, E)
    if lower_ok:
        Ql, Dl = closest_point_on_segment(P, E, Wr)

    if upper_ok and lower_ok:
        omega = 0.5 + 0.5 * np.clip((Dl - Du) / 8.0, -1.0, 1.0)
        D = np.minimum(Du, Dl)
        R = omega * UPPER_RADIUS + (1.0 - omega) * LOWER_RADIUS
        Q = np.where((Du <= Dl)[..., None], Qu, Ql)
    elif upper_ok:
        omega = np.ones((h, w), np.float32)
        D, Q = Du, Qu
        R = np.full((h, w), UPPER_RADIUS, np.float32)
    else:
        omega = np.zeros((h, w), np.float32)
        D, Q = Dl, Ql
        R = np.full((h, w), LOWER_RADIUS, np.float32)

    r = np.maximum(D - R, 0.0)
    W = weight_W(r, FEATHER)
    M_active = (D <= R + FEATHER).astype(np.float32)
    G = W * M_active

    return {
        'Q': Q.astype(np.float32),
        'G': G.astype(np.float32)
    }


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


# =====================================================================
# 1. ฟังก์ชัน Validate: ตรวจสอบแค่ว่าพบแขนข้างใดข้างหนึ่งหรือไม่ (คืนค่า 1 หรือ 0)
# =====================================================================
def validate_arm(image_matrix: np.ndarray) -> int:
    """
    ตรวจสอบความเห็นชัดของแขนในภาพ
    Return: 1 ถ้าพบแขนอย่างน้อย 1 ข้าง, 0 ถ้าไม่พบ
    """
    if image_matrix is None:
        return 0

    try:
        img_rgb = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2RGB)
        img_rgb = np.ascontiguousarray(img_rgb)

        landmarker = get_pose_landmarker()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return 0

        lm = result.pose_landmarks[0]
        arms = [
            (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
            (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
        ]

        for i_s, i_e, i_w in arms:
            vis = [lm[i].visibility for i in (i_s, i_e, i_w)]
            upper_ok = (vis[0] >= VISIBILITY_THRESHOLD and vis[1] >= VISIBILITY_THRESHOLD)
            lower_ok = (vis[1] >= VISIBILITY_THRESHOLD and vis[2] >= VISIBILITY_THRESHOLD)
            if upper_ok or lower_ok:
                return 1

        return 0
    except Exception as e:
        print(f"[ERROR validate_arm]: {e}")
        return 0


# =====================================================================
# 2. ฟังก์ชัน Masking: คำนวณสร้าง Mask และข้อมูล Field เก็บเข้า Cache
# =====================================================================
def create_arm_mask(image_matrix: np.ndarray):
    """
    สร้าง Mask และดึงแกนคำนวณแขนเตรียมเก็บเข้า Cache
    Return: (M_display, arm_cache_data) หรือ (None, None)
    """
    if image_matrix is None:
        return None, None

    h, w = image_matrix.shape[:2]
    img_rgb = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2RGB)
    img_rgb = np.ascontiguousarray(img_rgb)

    landmarker = get_pose_landmarker()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = landmarker.detect(mp_image)

    if not result.pose_landmarks or not result.segmentation_masks:
        return None, None

    lm = result.pose_landmarks[0]

    def pt(idx):
        return np.array([lm[idx].x * w, lm[idx].y * h], dtype=np.float32)

    # Person Mask
    seg_mask = result.segmentation_masks[0].numpy_view()
    M_person = (seg_mask >= 0.30).astype(np.float32)
    k = 2 * FEATHER + 1
    M_person = cv2.dilate(M_person, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    M_person = np.clip(cv2.GaussianBlur(M_person, (0, 0), FEATHER / 3), 0, 1)

    # Torso Protection
    torso = np.array([
        pt(LEFT_SHOULDER), pt(RIGHT_SHOULDER),
        pt(RIGHT_HIP), pt(LEFT_HIP)
    ], dtype=np.int32)

    M_torso = np.zeros((h, w), np.float32)
    cv2.fillPoly(M_torso, [torso], 1.0)
    M_torso = cv2.dilate(M_torso, np.ones((7, 7), np.uint8))
    M_torso = np.clip(cv2.GaussianBlur(M_torso, (7, 7), 0), 0, 1)

    arms = [
        (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
        (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
    ]

    P = pixel_grid(h, w)
    fields = []
    G_combined = np.zeros((h, w), dtype=np.float32)

    for i_s, i_e, i_w in arms:
        vis = [lm[i].visibility for i in (i_s, i_e, i_w)]
        upper_ok = (vis[0] >= VISIBILITY_THRESHOLD and vis[1] >= VISIBILITY_THRESHOLD)
        lower_ok = (vis[1] >= VISIBILITY_THRESHOLD and vis[2] >= VISIBILITY_THRESHOLD)

        if not (upper_ok or lower_ok):
            continue

        S, E, Wr = pt(i_s), pt(i_e), pt(i_w)
        field = build_arm_field(h, w, S, E, Wr, upper_ok, lower_ok)

        G_final = field['G'] * M_person * (1.0 - M_torso)
        field['G'] = G_final.astype(np.float32)

        if np.any(G_final > 1e-4):
            fields.append(field)
            G_combined += G_final

    if not fields:
        return None, None

    M_display = (np.clip(G_combined, 0.0, 1.0) * 255).astype(np.uint8)

    arm_cache_data = {
        'fields': fields,
        'P': P
    }

    return M_display, arm_cache_data


# =====================================================================
# 3. ฟังก์ชัน Adjust: ใช้ arm_cache_data กับค่า scale_val ปรับแต่งภาพ
# =====================================================================
def arm_adjust(image_matrix: np.ndarray, arm_cache_data: dict, scale_val: float = 0.0):
    """
    ประมวลผลปรับขนาดแขนด้วย Vector Scale Matrix แบบรวดเร็ว
    """
    scale = 1.0 - (scale_val / 100.0)
    if abs(scale - 1.0) < 1e-4:
        return image_matrix.copy()

    fields = arm_cache_data['fields']
    P = arm_cache_data['P']
    V_total = np.zeros_like(P, dtype=np.float32)

    for field in fields:
        Q = field['Q']
        G = field['G']
        s = np.maximum(1.0 + (scale - 1.0) * G, 0.05).astype(np.float32)

        inv_s = 1.0 / s
        delta = P - Q
        delta_source = delta * inv_s[..., None]
        source = Q + delta_source
        V_total += (source - P)

    source_all = P + V_total
    return bilinear_sample(image_matrix, source_all[..., 0], source_all[..., 1])