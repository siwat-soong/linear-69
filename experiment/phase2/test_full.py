"""
ไม่ยากถ้าอยากผอม : บีบ/ขยายแขนซ้าย-ขวาด้วย Vector-Matrix

Pipeline
  1. กำหนดภาพในโค้ด
  2. MediaPipe : หา landmark + person mask เท่านั้น
  3. สร้าง mask แขน  : vector projection + Euclidean distance
  4. Hadamard Product : ตัดฉากหลัง / ลำตัว
  5. Local scaling + inverse mapping + bilinear interpolation
  6. Slider เดียว  -50 .. +50  (ปรับแขนซ้ายและขวาพร้อมกัน)
        -50 = แขนใหญ่ขึ้น,  0 = ปกติ,  +50 = แขนเล็กลง

หมายเหตุ: OpenCV trackbar ติดลบไม่ได้ ตัว slider จึงแสดงเลข 0..100
          โดย 50 = ค่า 0  (ค่าจริง -50..+50 แสดงบนภาพ)
"""

import cv2
import numpy as np
import mediapipe as mp


# ============================================================
# 1. ตั้งค่า
# ============================================================
IMAGE_PATH = './dataset/img_0034_jpeg.jpg'

UPPER_RADIUS = 22        # รัศมีต้นแขน (px)
LOWER_RADIUS = 18        # รัศมีปลายแขน (px)
FEATHER = 25             # รัศมีอิทธิพล R ของ W(r) นอกตัวแขน (px)
VISIBILITY_THRESHOLD = 0.65


# ============================================================
# 2. Vector projection + Euclidean distance
# ============================================================
def pixel_grid(h, w):
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32),
                       np.arange(h, dtype=np.float32))
    return np.stack([X, Y], axis=-1)                 # (H, W, 2)


def closest_point_on_segment(P, A, B):
    """
    t = (AP · AB) / (AB · AB)      projection ลงบนเส้น AB
    t = clip(t, 0, 1)              จำกัดให้อยู่บน segment
    Q = A + t·AB                   จุดบนแกนแขนที่ใกล้ P ที่สุด
    D = ||P - Q||                  Euclidean distance
    """
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    AB = B - A
    L2 = float(AB @ AB)

    if L2 < 1e-6:
        Q = np.broadcast_to(A, P.shape).copy()
    else:
        t = np.clip(((P - A) @ AB) / L2, 0.0, 1.0)
        Q = A + t[..., None] * AB

    D = np.linalg.norm(P - Q, axis=-1)
    return Q.astype(np.float32), D.astype(np.float32)


def weight_W(r, R):
    """W(r) = (1 - r²/R²)²  เมื่อ r <= R,  0 เมื่อ r > R  (ลดลงต่อเนื่อง)"""
    q = np.clip(r / R, 0.0, 1.0)
    return (1.0 - q ** 2) ** 2


# ============================================================
# 3. Mask ของแขนหนึ่งข้าง + จุดบนแกนแขน Q
# ============================================================
def build_arm_field(h, w, S, E, Wr, upper_ok, lower_ok):
    """
    คืนค่า
      Q : (H,W,2)  จุดบนแกนแขนที่ใกล้แต่ละพิกเซลที่สุด
      G : (H,W)    น้ำหนักการบิด  = W(r) ⊙ M_active
                   (ในตัวแขน = 1, ออกนอกแขนค่อย ๆ ลดเป็น 0)
    """
    P = pixel_grid(h, w)

    if upper_ok:
        Qu, Du = closest_point_on_segment(P, S, E)
    if lower_ok:
        Ql, Dl = closest_point_on_segment(P, E, Wr)

    if upper_ok and lower_ok:
        # ω ผสมต้นแขน/ปลายแขนอย่างต่อเนื่องรอบข้อศอก (ไม่ให้รัศมีกระโดด)
        omega = 0.5 + 0.5 * np.clip((Dl - Du) / 8.0, -1.0, 1.0)
        D = np.minimum(Du, Dl)
        R = omega * UPPER_RADIUS + (1.0 - omega) * LOWER_RADIUS
        Q = np.where((Du <= Dl)[..., None], Qu, Ql)
    elif upper_ok:
        D, Q = Du, Qu
        R = np.full((h, w), UPPER_RADIUS, np.float32)
    else:
        D, Q = Dl, Ql
        R = np.full((h, w), LOWER_RADIUS, np.float32)

    r = np.maximum(D - R, 0.0)                       # ระยะส่วนเกินนอกแขน
    W = weight_W(r, FEATHER)
    M_active = (D <= R + FEATHER).astype(np.float32)

    return Q, (W * M_active).astype(np.float32)


# ============================================================
# 4. Bilinear interpolation
#    f = (1-wy)[(1-wx) f00 + wx f10] + wy[(1-wx) f01 + wx f11]
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
# 5. Local scaling + inverse mapping
# ============================================================
def reshape_arms(img, P, fields, scale):
    """
    scale < 1 = แขนเล็กลง,  scale > 1 = แขนใหญ่ขึ้น

    ตัวคูณ local ของแต่ละพิกเซล      s(P) = 1 + (scale - 1) · G(P)
    inverse mapping รอบจุดบนแกนแขน   source = Q + (P - Q) / s(P)
    ผลต่าง (displacement)            V = source - P
    แขนหลายข้างรวม V แบบเวกเตอร์    P_source = P + Σ V_i
    """
    if abs(scale - 1.0) < 1e-4:
        return img.copy()

    V_total = np.zeros_like(P)

    for Q, G in fields:
        s = np.maximum(1.0 + (scale - 1.0) * G, 0.05)
        source = Q + (P - Q) / s[..., None]
        V_total += source - P

    source_all = P + V_total
    return bilinear_sample(img, source_all[..., 0], source_all[..., 1])


# ============================================================
# 6. โปรแกรมหลัก
# ============================================================
def main():
    # ---------- กำหนดภาพ ----------
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("ไม่พบไฟล์รูปภาพ")
        return

    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ---------- MediaPipe : landmark + person mask เท่านั้น ----------
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True,
                      model_complexity=1,
                      enable_segmentation=True) as pose:
        result = pose.process(img_rgb)

    if not result.pose_landmarks or result.segmentation_mask is None:
        print("ไม่พบ pose landmarks / person mask")
        return

    lm = result.pose_landmarks.landmark
    L = mp_pose.PoseLandmark

    def pt(idx):
        return np.array([lm[idx].x * w, lm[idx].y * h], dtype=np.float32)

    # ---------- M_person : ขยายและเกลี่ยขอบให้กว้างกว่าตัวคนเล็กน้อย ----------
    # (ถ้าใช้ mask ตัวคนแบบพอดีตัว การบิดจะถูกตัดที่ขอบแขนแล้วเห็นรอยต่อ
    #  จึงขยายออกอย่างน้อยเท่า FEATHER เพื่อตัดเฉพาะฉากหลังที่ไกลออกไป)
    M_person = (result.segmentation_mask >= 0.30).astype(np.float32)
    k = 2 * FEATHER + 1
    M_person = cv2.dilate(M_person,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    M_person = np.clip(cv2.GaussianBlur(M_person, (0, 0), FEATHER / 3), 0, 1)

    # ---------- M_torso : ลำตัว (ห้ามบิด) ----------
    torso = np.array([pt(L.LEFT_SHOULDER), pt(L.RIGHT_SHOULDER),
                      pt(L.RIGHT_HIP), pt(L.LEFT_HIP)], dtype=np.int32)
    M_torso = np.zeros((h, w), np.float32)
    cv2.fillPoly(M_torso, [torso], 1.0)
    M_torso = cv2.dilate(M_torso, np.ones((7, 7), np.uint8))
    M_torso = np.clip(cv2.GaussianBlur(M_torso, (7, 7), 0), 0, 1)

    # ---------- หาแขนซ้าย / ขวา ----------
    arms = {
        'left':  (L.LEFT_SHOULDER,  L.LEFT_ELBOW,  L.LEFT_WRIST),
        'right': (L.RIGHT_SHOULDER, L.RIGHT_ELBOW, L.RIGHT_WRIST),
    }

    P = pixel_grid(h, w)
    fields = []
    points = []

    for name, (i_s, i_e, i_w) in arms.items():
        vis = [lm[i].visibility for i in (i_s, i_e, i_w)]
        upper_ok = vis[0] >= VISIBILITY_THRESHOLD and vis[1] >= VISIBILITY_THRESHOLD
        lower_ok = vis[1] >= VISIBILITY_THRESHOLD and vis[2] >= VISIBILITY_THRESHOLD

        print(f"{name}: visibility={[round(v, 3) for v in vis]} "
              f"upper={'detect' if upper_ok else 'skip'} "
              f"lower={'detect' if lower_ok else 'skip'}")

        if not (upper_ok or lower_ok):
            continue

        S, E, Wr = pt(i_s), pt(i_e), pt(i_w)
        Q, G = build_arm_field(h, w, S, E, Wr, upper_ok, lower_ok)

        # Hadamard Product : ตัดฉากหลัง + ตัดลำตัว
        #   G_final = G ⊙ M_person ⊙ (1 - M_torso)
        G = G * M_person * (1.0 - M_torso)

        fields.append((Q, G))
        points.append((S, E, Wr, upper_ok, lower_ok))

    if not fields:
        print("มองไม่เห็นแขนทั้งสองข้าง")
        return

    # ---------- แสดง mask ที่ใช้บิด ----------
    G_all = np.clip(np.maximum.reduce([G for _, G in fields]), 0, 1)
    overlay = img.copy()
    red = np.zeros_like(img)
    red[:, :, 2] = (G_all * 255).astype(np.uint8)
    overlay = cv2.addWeighted(overlay, 1.0, red, 0.5, 0)

    for S, E, Wr, u_ok, l_ok in points:
        if u_ok:
            cv2.line(overlay, tuple(S.astype(int)), tuple(E.astype(int)),
                     (255, 0, 0), 2)
        if l_ok:
            cv2.line(overlay, tuple(E.astype(int)), tuple(Wr.astype(int)),
                     (255, 0, 0), 2)
        for p in (S, E, Wr):
            cv2.circle(overlay, tuple(p.astype(int)), 5, (0, 255, 0), -1)

    cv2.imshow("1 - Original", img)
    cv2.imshow("2 - Deform Mask", overlay)

    # ---------- Slider  -50 .. +50 ----------
    WIN = "3 - Result"
    BAR = "slider (50 = normal)"
    cv2.namedWindow(WIN)
    cv2.createTrackbar(BAR, WIN, 50, 100, lambda v: None)

    last = None
    while True:
        slider = cv2.getTrackbarPos(BAR, WIN) - 50       # -50 .. +50

        if slider != last:
            # +50 -> scale 0.5 (เล็กลง),  0 -> 1.0,  -50 -> scale 1.5 (ใหญ่ขึ้น)
            scale = 1.0 - slider / 100.0
            out = reshape_arms(img, P, fields, scale)

            cv2.putText(out, f"slider = {slider:+d}   scale = {scale:.2f}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)
            cv2.imshow(WIN, out)
            last = slider

        if cv2.waitKey(30) & 0xFF == 27:
            break
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()