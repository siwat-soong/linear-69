# โค้ดส่วนกลางที่ใช้ร่วมกันของ chest.py / waist.py / hips.py
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


POSE_MODEL_PATH = './models/pose_landmarker_full.task'

# mediapipe รุ่นใหม่บน macOS (Apple Silicon) ไม่มี legacy `mp.solutions.pose` แล้ว
# เหลือแต่ Tasks API (`mp.tasks`) จึงกำหนด index ของ landmark เอง (BlazePose 33 จุด
# มี index เดิมเหมือนกันทุกเวอร์ชัน ไม่ว่าจะเรียกผ่าน API ไหน)
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24

VISIBILITY_THRESHOLD = 0.65
POLY_FEATHER = 25   # px, เกลี่ยขอบ polygon ลำตัว


# ============================================================
# Vector projection + Euclidean distance (ใช้ร่วมกับแขน/ขา)
# ============================================================
def pixel_grid(h, w):
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32),
                       np.arange(h, dtype=np.float32))
    return np.stack([X, Y], axis=-1)                 # (H, W, 2)


def project_on_axis(P, A, B):
    """
    t = (AP · AB) / (AB · AB)      projection ลงบนแกน AB (ไม่ clip)
    Q = A + clip(t,0,1)·AB         จุดบนแกนที่ใกล้ P ที่สุด (อยู่บน segment)
    """
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    AB = B - A
    L2 = float(AB @ AB)

    t_raw = ((P - A) @ AB) / max(L2, 1e-6)
    t_clip = np.clip(t_raw, 0.0, 1.0)
    Q = A + t_clip[..., None] * AB
    return Q.astype(np.float32), t_raw.astype(np.float32)


def gaussian_band(t, t_center, sigma):
    """น้ำหนัก Gaussian ตามตำแหน่งบนแกน t (0=ไหล่ .. 1=สะโพก)"""
    return np.exp(-((t - t_center) ** 2) / (2.0 * sigma ** 2))


def scale_from_slider(scale_val):
    """scale_val -50..+50  ->  s = 1 - scale_val/100
       scale_val > 0 (เลื่อนขวา) = บีบเล็กลง,  scale_val < 0 = ขยายใหญ่ขึ้น"""
    return 1.0 - scale_val / 100.0


# ============================================================
# Mask ลำตัวด้วย Point-in-Polygon (Vector Cross Product Determinant)
# ============================================================
def point_in_polygon_mask(h, w, poly_pts):
    """
    ใช้เครื่องหมาย cross product ของเวกเตอร์ขอบ (edge) เทียบกับเวกเตอร์จากจุดขอบ
    ไปยังพิกเซล เพื่อตรวจว่าพิกเซลอยู่ "ฝั่งเดียวกัน" ของทุกขอบหรือไม่
    (polygon นูน 4 จุด: ไหล่ซ้าย -> ไหล่ขวา -> สะโพกขวา -> สะโพกซ้าย)

        cross(edge, P - V_i) มีเครื่องหมายเดียวกันทุกขอบ  <=>  P อยู่ในรูปหลายเหลี่ยม
    """
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

    M_poly = (np.abs(sign_sum) >= n - 0.5).astype(np.float32)  # เครื่องหมายเดียวกันทุกขอบ
    return M_poly


def build_torso_polygon_mask(h, w, L_sh, R_sh, R_hip, L_hip):
    M_hard = point_in_polygon_mask(h, w, [L_sh, R_sh, R_hip, L_hip])
    k = 2 * POLY_FEATHER + 1
    M_soft = cv2.dilate(M_hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    M_soft = np.clip(cv2.GaussianBlur(M_soft, (0, 0), POLY_FEATHER / 3), 0, 1)
    return M_soft


# ============================================================
# Bilinear interpolation
# ============================================================
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
# Local scaling (ตั้งฉากแกนกลางลำตัวเท่านั้น) + inverse mapping
# ทีละ 1 ส่วน (อก หรือ เอว หรือ สะโพก อย่างใดอย่างหนึ่ง) เพื่อให้ปรับแยกกันได้จริง
# ============================================================
def reshape_single_part(img, P, Q, t_raw, G_body, t_center, sigma, scale_val):
    """
    บีบ/ขยายเฉพาะ "ส่วนเดียว" ตามแถบ Gaussian รอบ t_center (อก หรือ เอว หรือ สะโพก)
    ส่วนอื่นของลำตัวไม่ถูกแตะต้อง (G_band ของพิกเซลนอกแถบ ~ 0)

    G_band(P) = Gaussian(t, t_center, sigma) ⊙ M_torso(P) ⊙ M_person(P)
    s(P)      = 1 + (scale - 1) · G_band(P)
    source    = Q + (P - Q) / s(P)      <- บีบเฉพาะทิศตั้งฉากแกนกลางลำตัว
    """
    scale = scale_from_slider(scale_val)
    if abs(scale - 1.0) < 1e-4:
        return img.copy()

    G_band = gaussian_band(t_raw, t_center, sigma) * G_body
    s_field = 1.0 + (scale - 1.0) * G_band
    s_field = np.maximum(s_field, 0.05)

    perp = P - Q                              # เวกเตอร์ตั้งฉากแกนกลาง (ทิศบีบ/ขยาย)
    source = Q + perp / s_field[..., None]    # แกน t (แนวดิ่งลำตัว) ไม่ขยับ

    return bilinear_sample(img, source[..., 0], source[..., 1])


# ============================================================
# ตรวจจับ Pose + Segmentation Mask ด้วย Tasks API
# ============================================================
def detect_pose(img_rgb):
    """คืนค่า (landmarks, segmentation_mask) หรือ (None, None) ถ้าไม่เจอคน"""
    base_options = BaseOptions(model_asset_path=POSE_MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
    )
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        return None, None

    landmarks = result.pose_landmarks[0]                 # คนแรกที่เจอ
    seg_mask = result.segmentation_masks[0].numpy_view()  # (H, W) float32 0..1
    return landmarks, seg_mask


# ============================================================
# เตรียมข้อมูลร่วม (ใช้ซ้ำได้ทั้ง 3 ไฟล์ปรับแต่ง + สคริปต์คำนวณมือ)
# ============================================================
def prepare_body_data(image_path):
    """
    โหลดภาพ -> หา pose landmark -> คำนวณแกนกลางลำตัว Sc->Hc -> สร้าง G_body
    คืนค่า dict ที่มีทุกอย่างที่ต้องใช้ต่อ (รวม Q, t_raw ของทุกพิกเซลในภาพ)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"ไม่พบไฟล์ภาพที่ '{image_path}'")

    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    lm, seg_mask = detect_pose(img_rgb)
    if lm is None:
        raise RuntimeError("ไม่พบ pose landmarks / person mask ในภาพนี้")

    def pt(idx):
        return np.array([lm[idx].x * w, lm[idx].y * h], dtype=np.float32)

    vis_needed = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
    if min(lm[i].visibility for i in vis_needed) < VISIBILITY_THRESHOLD:
        raise RuntimeError("มองไม่เห็นไหล่หรือสะโพกชัดเจนพอในภาพนี้")

    L_sh, R_sh = pt(LEFT_SHOULDER), pt(RIGHT_SHOULDER)
    L_hip, R_hip = pt(LEFT_HIP), pt(RIGHT_HIP)

    Sc = (L_sh + R_sh) / 2.0    # จุดกึ่งกลางไหล่ (A)
    Hc = (L_hip + R_hip) / 2.0  # จุดกึ่งกลางสะโพก (B)

    # ---------- M_person : กันฉากหลัง ----------
    M_person = (seg_mask >= 0.30).astype(np.float32)
    k = 2 * POLY_FEATHER + 1
    M_person = cv2.dilate(M_person,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    M_person = np.clip(cv2.GaussianBlur(M_person, (0, 0), POLY_FEATHER / 3), 0, 1)

    # ---------- M_torso : polygon ไหล่-สะโพก (Cross Product Determinant) ----------
    M_torso = build_torso_polygon_mask(h, w, L_sh, R_sh, R_hip, L_hip)
    G_body = M_torso * M_person

    # ---------- โปรเจกต์ทุกพิกเซลลงบนแกนกลางลำตัว ----------
    P = pixel_grid(h, w)
    Q, t_raw = project_on_axis(P, Sc, Hc)

    return {
        'img': img, 'h': h, 'w': w, 'P': P, 'Q': Q, 't_raw': t_raw,
        'G_body': G_body, 'M_torso': M_torso, 'M_person': M_person,
        'L_sh': L_sh, 'R_sh': R_sh, 'L_hip': L_hip, 'R_hip': R_hip,
        'Sc': Sc, 'Hc': Hc,
    }
