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

# Index ของ Landmarks
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

# พารามิเตอร์ขนาด Mask ขาและเกราะป้องกัน
THIGH_RADIUS = 32
CALF_RADIUS = 24
LEG_FEATHER = 28

ARM_UPPER_RADIUS = 24
ARM_LOWER_RADIUS = 20
HAND_EXTRA_RADIUS = 14

VISIBILITY_THRESHOLD = 0.35  # ปรับลดความเข้มงวดเพื่อให้ตรวจจับในสภาพแสง/มุมเอียงจริงได้ง่ายขึ้น

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


def build_single_segment_field(h, w, A, B, radius, feather):
    P = pixel_grid(h, w)
    Q, D = closest_point_on_segment(P, A, B)
    r = np.maximum(D - radius, 0.0)
    W = weight_W(r, feather)
    M_active = (D <= radius + feather).astype(np.float32)
    G = W * M_active
    return Q.astype(np.float32), G.astype(np.float32)


def build_arm_protection_mask(h, w, shoulder, elbow, wrist, upper_ok, lower_ok):
    P = pixel_grid(h, w)
    M_upper = np.zeros((h, w), dtype=np.float32)
    M_lower = np.zeros((h, w), dtype=np.float32)

    if upper_ok:
        _, D_upper = closest_point_on_segment(P, shoulder, elbow)
        M_upper = (D_upper <= ARM_UPPER_RADIUS).astype(np.float32)

    if lower_ok:
        _, D_lower = closest_point_on_segment(P, elbow, wrist)
        M_lower = (D_lower <= ARM_LOWER_RADIUS).astype(np.float32)

    M_arm = np.maximum(M_upper, M_lower)

    if lower_ok:
        X, Y = P[..., 0], P[..., 1]
        D_hand = np.sqrt((X - wrist[0]) ** 2 + (Y - wrist[1]) ** 2)
        hand_radius = ARM_LOWER_RADIUS + HAND_EXTRA_RADIUS
        M_hand = (D_hand <= hand_radius).astype(np.float32)
        M_arm = np.maximum(M_arm, M_hand)

    kernel = np.ones((7, 7), dtype=np.uint8)
    M_arm = cv2.dilate(M_arm, kernel, iterations=1)
    M_arm = cv2.GaussianBlur(M_arm, (9, 9), 0)
    return np.clip(M_arm, 0.0, 1.0)


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
# 1. ฟังก์ชัน Validate: ตรวจสอบแค่ว่าพบต้นขาหรือน่องอย่างน้อย 1 ข้างหรือไม่ (คืนค่า 1 หรือ 0)
# =====================================================================
def validate_leg(image_matrix: np.ndarray) -> int:
    """
    ตรวจสอบความเห็นชัดของต้นขาหรือน่องในภาพ
    Return: 1 ถ้าพบขาอย่างน้อย 1 ส่วน, 0 ถ้าไม่พบ
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
        legs = [
            (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
            (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
        ]

        for hip_id, knee_id, ankle_id in legs:
            hip_vis = lm[hip_id].visibility
            knee_vis = lm[knee_id].visibility
            ankle_vis = lm[ankle_id].visibility

            thigh_ok = (hip_vis >= VISIBILITY_THRESHOLD and knee_vis >= VISIBILITY_THRESHOLD)
            calf_ok = (knee_vis >= VISIBILITY_THRESHOLD and ankle_vis >= VISIBILITY_THRESHOLD)

            if thigh_ok or calf_ok:
                return 1

        return 0
    except Exception as e:
        print(f"[ERROR validate_leg]: {e}")
        return 0


# =====================================================================
# 2. ฟังก์ชัน Masking: คำนวณสร้าง Mask และข้อมูล Field เก็บเข้า Cache
# =====================================================================
def create_leg_mask(image_matrix: np.ndarray):
    """
    สร้าง Mask และคำนวณ Field แยกต้นขา/น่องเตรียมเก็บเข้า Cache
    Return: (M_display, leg_cache_data) หรือ (None, None)
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
    k = 2 * LEG_FEATHER + 1
    M_person = cv2.dilate(M_person, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    M_person = cv2.GaussianBlur(M_person, (0, 0), LEG_FEATHER / 3)
    M_person = np.clip(M_person, 0.0, 1.0)

    # Torso Protection
    torso_pts = np.array([
        pt(LEFT_SHOULDER), pt(RIGHT_SHOULDER),
        pt(RIGHT_HIP), pt(LEFT_HIP)
    ], dtype=np.int32)

    M_torso = np.zeros((h, w), dtype=np.float32)
    cv2.fillPoly(M_torso, [torso_pts], 1.0)
    M_torso = cv2.dilate(M_torso, np.ones((7, 7), dtype=np.uint8), iterations=1)
    M_torso = cv2.GaussianBlur(M_torso, (7, 7), 0)
    M_torso = np.clip(M_torso, 0.0, 1.0)

    # Arm Protection Mask
    arm_defs = [
        (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
        (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
    ]
    M_arm_protect = np.zeros((h, w), dtype=np.float32)

    for sh_id, el_id, wr_id in arm_defs:
        sh_vis, el_vis, wr_vis = lm[sh_id].visibility, lm[el_id].visibility, lm[wr_id].visibility
        upper_ok = (sh_vis >= VISIBILITY_THRESHOLD and el_vis >= VISIBILITY_THRESHOLD)
        lower_ok = (el_vis >= VISIBILITY_THRESHOLD and wr_vis >= VISIBILITY_THRESHOLD)

        if upper_ok or lower_ok:
            M_one_arm = build_arm_protection_mask(h, w, pt(sh_id), pt(el_id), pt(wr_id), upper_ok, lower_ok)
            M_arm_protect = np.maximum(M_arm_protect, M_one_arm)

    M_arm_protect = np.clip(M_arm_protect, 0.0, 1.0)

    # Leg Fields (ต้นขา + น่อง)
    legs = [
        (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
        (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
    ]

    P = pixel_grid(h, w)
    fields = []
    G_combined = np.zeros((h, w), dtype=np.float32)

    for hip_id, knee_id, ankle_id in legs:
        hip_vis, knee_vis, ankle_vis = lm[hip_id].visibility, lm[knee_id].visibility, lm[ankle_id].visibility
        thigh_ok = (hip_vis >= VISIBILITY_THRESHOLD and knee_vis >= VISIBILITY_THRESHOLD)
        calf_ok = (knee_vis >= VISIBILITY_THRESHOLD and ankle_vis >= VISIBILITY_THRESHOLD)

        if not (thigh_ok or calf_ok):
            continue

        H, K, A = pt(hip_id), pt(knee_id), pt(ankle_id)

        # Thigh Field
        if thigh_ok:
            Q_thigh, G_thigh = build_single_segment_field(h, w, H, K, THIGH_RADIUS, LEG_FEATHER)
            G_thigh = G_thigh * M_person * (1.0 - M_torso) * (1.0 - M_arm_protect)
            G_thigh = np.clip(cv2.GaussianBlur(G_thigh, (7, 7), 0), 0.0, 1.0)

            if np.any(G_thigh > 1e-4):
                fields.append((Q_thigh, G_thigh))
                G_combined += G_thigh

        # Calf Field
        if calf_ok:
            Q_calf, G_calf = build_single_segment_field(h, w, K, A, CALF_RADIUS, LEG_FEATHER)
            G_calf = G_calf * M_person * (1.0 - M_torso) * (1.0 - M_arm_protect)
            G_calf = np.clip(cv2.GaussianBlur(G_calf, (7, 7), 0), 0.0, 1.0)

            if np.any(G_calf > 1e-4):
                fields.append((Q_calf, G_calf))
                G_combined += G_calf

    if not fields:
        return None, None

    M_display = (np.clip(G_combined, 0.0, 1.0) * 255).astype(np.uint8)

    leg_cache_data = {
        'fields': fields,
        'P': P
    }

    return M_display, leg_cache_data


# =====================================================================
# 3. ฟังก์ชัน Adjust: ใช้ leg_cache_data กับค่า scale_val ปรับแต่งภาพ
# =====================================================================
def leg_adjust(image_matrix: np.ndarray, leg_cache_data: dict, scale_val: float = 0.0):
    """
    ประมวลผลปรับขนาดต้นขาและน่องด้วย Inverse Mapping แบบรวดเร็ว
    """
    scale = 1.0 - (scale_val / 100.0)
    if abs(scale - 1.0) < 1e-4:
        return image_matrix.copy()

    fields = leg_cache_data['fields']
    P = leg_cache_data['P']
    V_total = np.zeros_like(P, dtype=np.float32)

    for Q, G in fields:
        s = np.maximum(1.0 + (scale - 1.0) * G, 0.05).astype(np.float32)
        source = Q + (P - Q) / s[..., None]
        V_total += (source - P)

    source_all = P + V_total
    return bilinear_sample(image_matrix, source_all[..., 0], source_all[..., 1])