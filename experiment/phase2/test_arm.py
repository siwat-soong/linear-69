import cv2
import numpy as np
import mediapipe as mp


# ============================================================
# 1) หาระยะจากทุก pixel ไปยังเส้นช่วง AB
# ============================================================
def point_to_segment_distance_map(height, width, A, B):
    X, Y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32)
    )

    P = np.stack([X, Y], axis=-1)

    A = np.array(A, dtype=np.float32)
    B = np.array(B, dtype=np.float32)

    AB = B - A
    AB_len_sq = np.sum(AB ** 2)

    # กรณี A กับ B ซ้อนกัน
    if AB_len_sq < 1e-6:
        diff = P - A
        return np.sqrt(
            np.sum(diff ** 2, axis=-1)
        )

    AP = P - A

    # t = (AP dot AB) / (AB dot AB)
    t = np.sum(
        AP * AB,
        axis=-1
    ) / AB_len_sq

    # จำกัด projection ให้อยู่บน segment AB
    t = np.clip(
        t,
        0.0,
        1.0
    )

    # จุด Q ที่ใกล้ pixel ที่สุดบน AB
    Q = (
        A +
        t[..., None] * AB
    )

    diff = P - Q

    dist = np.sqrt(
        np.sum(
            diff ** 2,
            axis=-1
        )
    )

    return dist


# ============================================================
# 2) สร้าง Soft Capsule Mask
# ============================================================
def capsule_soft_mask(
    height,
    width,
    A,
    B,
    radius,
    sigma=10.0
):
    dist = point_to_segment_distance_map(
        height,
        width,
        A,
        B
    )

    mask = np.where(
        dist <= radius,
        1.0,
        np.exp(
            -((dist - radius) ** 2)
            /
            (2.0 * sigma ** 2)
        )
    )

    return mask.astype(np.float32)


# ============================================================
# 3) หา Q = จุดบน segment ที่ใกล้ pixel ที่สุด
# ============================================================
def closest_point_on_segment_map(
    height,
    width,
    A,
    B
):
    X, Y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32)
    )

    P = np.stack(
        [X, Y],
        axis=-1
    )

    A = np.array(A, dtype=np.float32)
    B = np.array(B, dtype=np.float32)

    AB = B - A
    AB_len_sq = np.sum(AB ** 2)

    if AB_len_sq < 1e-6:
        Q = np.zeros_like(P)
        Q[:, :, 0] = A[0]
        Q[:, :, 1] = A[1]

        dist = np.sqrt(
            np.sum(
                (P - Q) ** 2,
                axis=-1
            )
        )

        return Q, dist

    AP = P - A

    t = (
        np.sum(
            AP * AB,
            axis=-1
        )
        / AB_len_sq
    )

    t = np.clip(
        t,
        0.0,
        1.0
    )

    Q = (
        A +
        t[..., None] * AB
    )

    dist = np.sqrt(
        np.sum(
            (P - Q) ** 2,
            axis=-1
        )
    )

    return Q, dist


# ============================================================
# 4) ฟังก์ชันบีบ / ขยายแขน
# ============================================================
def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def resize_arm_thickness(
    image,
    shoulder,
    elbow,
    wrist,
    scale,
    upper_radius,
    lower_radius,
    feather=25,
    protect_mask=None
):
    h, w = image.shape[:2]

    X, Y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )
    P = np.stack([X, Y], axis=-1)

    Q_u, D_u = closest_point_on_segment_map(h, w, shoulder, elbow)
    Q_l, D_l = closest_point_on_segment_map(h, w, elbow, wrist)

    use_upper = D_u <= D_l
    Q = np.where(use_upper[..., None], Q_u, Q_l)
    D = np.minimum(D_u, D_l)
    R = np.where(use_upper, upper_radius, lower_radius)

    # น้ำหนัก 1 = บิดเต็มที่ (ในแขน), ค่อย ๆ ลดเป็น 0 นอกแขน
    weight = 1.0 - smoothstep(R, R + feather, D)

    # ไม่ให้บิดเข้าไปในลำตัว
    if protect_mask is not None:
        weight = weight * (1.0 - protect_mask)

    # เกลี่ย field ให้นุ่ม
    weight = cv2.GaussianBlur(weight.astype(np.float32), (0, 0), 5)

    # scale ที่เปลี่ยนไปตามตำแหน่ง
    s = 1.0 + (scale - 1.0) * weight
    s = np.maximum(s, 0.05)

    source = Q + (P - Q) / s[..., None]

    return cv2.remap(
        image,
        source[:, :, 0].astype(np.float32),
        source[:, :, 1].astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101
    )

# ============================================================
# 5) อ่านภาพ
# ============================================================
img = cv2.imread(
    './dataset/img_0131_jpeg.jpg'
)

if img is None:
    print("ไม่พบไฟล์รูปภาพ")
    exit()

height, width = img.shape[:2]

img_rgb = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# 6) MediaPipe Pose
# ============================================================
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    enable_segmentation=True
)

result = pose.process(
    img_rgb
)

if not result.pose_landmarks:
    print("ไม่พบ pose landmarks")
    pose.close()
    exit()

landmarks = result.pose_landmarks.landmark


# ============================================================
# 7) Person Segmentation Mask
# ============================================================
if result.segmentation_mask is None:
    print("ไม่พบ Person Segmentation Mask")
    pose.close()
    exit()

M_person = result.segmentation_mask.astype(
    np.float32
)

# ตัดค่าความมั่นใจต่ำ
M_person = np.where(
    M_person >= 0.30,
    M_person,
    0.0
).astype(np.float32)

# เกลี่ยขอบ
M_person = cv2.GaussianBlur(
    M_person,
    (7, 7),
    0
)

M_person = np.clip(
    M_person,
    0.0,
    1.0
)


# ============================================================
# 8) Torso Mask
# ============================================================
left_shoulder_torso = landmarks[
    mp_pose.PoseLandmark.LEFT_SHOULDER
]

right_shoulder_torso = landmarks[
    mp_pose.PoseLandmark.RIGHT_SHOULDER
]

left_hip = landmarks[
    mp_pose.PoseLandmark.LEFT_HIP
]

right_hip = landmarks[
    mp_pose.PoseLandmark.RIGHT_HIP
]


LS_torso = np.array(
    [
        left_shoulder_torso.x * width,
        left_shoulder_torso.y * height
    ],
    dtype=np.int32
)

RS_torso = np.array(
    [
        right_shoulder_torso.x * width,
        right_shoulder_torso.y * height
    ],
    dtype=np.int32
)

LH = np.array(
    [
        left_hip.x * width,
        left_hip.y * height
    ],
    dtype=np.int32
)

RH = np.array(
    [
        right_hip.x * width,
        right_hip.y * height
    ],
    dtype=np.int32
)


torso_polygon = np.array(
    [
        LS_torso,
        RS_torso,
        RH,
        LH
    ],
    dtype=np.int32
)


M_torso = np.zeros(
    (height, width),
    dtype=np.float32
)

cv2.fillPoly(
    M_torso,
    [torso_polygon],
    1.0
)


# ============================================================
# 9) Torso Safety Margin
# ============================================================
torso_binary = (
    M_torso > 0.5
).astype(np.uint8)

torso_kernel = np.ones(
    (7, 7),
    np.uint8
)

M_torso_safe = cv2.dilate(
    torso_binary,
    torso_kernel,
    iterations=1
).astype(np.float32)

M_torso_safe = cv2.GaussianBlur(
    M_torso_safe,
    (7, 7),
    0
)

M_torso_safe = np.clip(
    M_torso_safe,
    0.0,
    1.0
)


# ============================================================
# 10) Landmark แขนซ้าย
# ============================================================
left_shoulder = landmarks[
    mp_pose.PoseLandmark.LEFT_SHOULDER
]

left_elbow = landmarks[
    mp_pose.PoseLandmark.LEFT_ELBOW
]

left_wrist = landmarks[
    mp_pose.PoseLandmark.LEFT_WRIST
]


LS = np.array(
    [
        left_shoulder.x * width,
        left_shoulder.y * height
    ],
    dtype=np.float32
)

LE = np.array(
    [
        left_elbow.x * width,
        left_elbow.y * height
    ],
    dtype=np.float32
)

LW = np.array(
    [
        left_wrist.x * width,
        left_wrist.y * height
    ],
    dtype=np.float32
)


# ============================================================
# 11) Visibility
# ============================================================
visibility_threshold = 0.65

shoulder_visible = (
    left_shoulder.visibility
    >= visibility_threshold
)

elbow_visible = (
    left_elbow.visibility
    >= visibility_threshold
)

wrist_visible = (
    left_wrist.visibility
    >= visibility_threshold
)

print(
    "Left Shoulder:",
    round(left_shoulder.visibility, 3)
)

print(
    "Left Elbow:",
    round(left_elbow.visibility, 3)
)

print(
    "Left Wrist:",
    round(left_wrist.visibility, 3)
)


# ============================================================
# 12) Radius แขน
# ============================================================
upper_radius = 22
lower_radius = 18


# ============================================================
# 13) สร้าง Capsule Mask
# ============================================================
M_upper = np.zeros(
    (height, width),
    dtype=np.float32
)

M_lower = np.zeros(
    (height, width),
    dtype=np.float32
)


# ------------------------------------------------------------
# ต้นแขน
# ------------------------------------------------------------
if shoulder_visible and elbow_visible:

    M_upper = capsule_soft_mask(
        height,
        width,
        LS,
        LE,
        upper_radius,
        sigma=10.0
    )

    print("Detect: Left Upper Arm")

else:

    print("Skip: Left Upper Arm")


# ------------------------------------------------------------
# ปลายแขน
# ------------------------------------------------------------
if elbow_visible and wrist_visible:

    M_lower = capsule_soft_mask(
        height,
        width,
        LE,
        LW,
        lower_radius,
        sigma=10.0
    )

    print("Detect: Left Lower Arm")

else:

    print("Skip: Left Lower Arm")


# ============================================================
# 14) รวมต้นแขน + ปลายแขน
# ============================================================
M_arm = np.maximum(
    M_upper,
    M_lower
)

M_arm = cv2.GaussianBlur(
    M_arm,
    (11, 11),
    0
)


# ============================================================
# 15) Final Arm Mask
# ============================================================
M_arm_final = (
    M_arm
    * M_person
    * (1.0 - M_torso_safe)
)

M_arm_final = np.clip(
    M_arm_final,
    0.0,
    1.0
)


# ============================================================
# 16) Clean Deform Mask
#
# ตัวนี้ใช้ "บีบแขนจริง"
# ============================================================
M_arm_binary = (
    M_arm_final > 0.25
).astype(np.uint8)

erode_kernel = np.ones(
    (3, 3),
    np.uint8
)

M_arm_clean = cv2.erode(
    M_arm_binary,
    erode_kernel,
    iterations=1
).astype(np.float32)

M_arm_clean = cv2.GaussianBlur(
    M_arm_clean,
    (5, 5),
    0
)

M_arm_clean = np.clip(
    M_arm_clean,
    0.0,
    1.0
)


# ============================================================
# 17) Blend Mask
#
# ขยายออกจาก Deform Mask เล็กน้อย
# เพื่อช่วยคลุม edge เดิม
# ============================================================
blend_binary = (
    M_arm_clean > 0.20
).astype(np.uint8)

# เดิม 7x7 -> เพิ่มเป็น 11x11
blend_kernel = np.ones(
    (13, 13),
    np.uint8
)

M_arm_blend = cv2.dilate(
    blend_binary,
    blend_kernel,
    iterations=1
).astype(np.float32)

# เกลี่ยขอบให้ไม่เป็นเส้นแข็ง
M_arm_blend = cv2.GaussianBlur(
    M_arm_blend,
    (9, 9),
    0
)

# สำคัญ:
# ถึง Blend Mask จะขยายออก
# ก็ยังไม่อนุญาตให้ล้ำเข้า Torso
M_arm_blend = (
    M_arm_blend
    * (1.0 - M_torso_safe)
)

M_arm_blend = np.clip(
    M_arm_blend,
    0.0,
    1.0
)


# ============================================================
# 18) ค่า Scale สำหรับทดสอบ
# ============================================================

# 0.90 = เล็กลง 10%
# 0.80 = เล็กลง 20%
# 1.00 = ปกติ
# 1.20 = ใหญ่ขึ้น 20%

test_scale = 0.6



# ============================================================
# 19) Resize
# ============================================================
result_resize = resize_arm_thickness(
    img,
    LS, LE, LW,
    test_scale,
    upper_radius,
    lower_radius,
    feather=25,
    protect_mask=M_torso_safe
)

# ============================================================
# 20) เตรียมภาพแสดงผล
# ============================================================
M_person_display = (
    M_person * 255
).astype(np.uint8)

M_torso_display = (
    M_torso_safe * 255
).astype(np.uint8)

M_arm_final_display = (
    M_arm_final * 255
).astype(np.uint8)

M_arm_clean_display = (
    M_arm_clean * 255
).astype(np.uint8)

M_arm_blend_display = (
    M_arm_blend * 255
).astype(np.uint8)


# ============================================================
# 21) Overlay Deform Mask
# ============================================================
overlay = img.copy()

red_layer = np.zeros_like(
    img
)

red_layer[:, :, 2] = (
    M_arm_clean_display
)

overlay = cv2.addWeighted(
    overlay,
    1.0,
    red_layer,
    0.5,
    0
)


# ============================================================
# 22) วาด Landmark
# ============================================================
if shoulder_visible:

    cv2.circle(
        overlay,
        tuple(LS.astype(int)),
        6,
        (0, 255, 0),
        -1
    )


if elbow_visible:

    cv2.circle(
        overlay,
        tuple(LE.astype(int)),
        6,
        (0, 255, 0),
        -1
    )


if wrist_visible:

    cv2.circle(
        overlay,
        tuple(LW.astype(int)),
        6,
        (0, 255, 0),
        -1
    )


if shoulder_visible and elbow_visible:

    cv2.line(
        overlay,
        tuple(LS.astype(int)),
        tuple(LE.astype(int)),
        (255, 0, 0),
        2
    )


if elbow_visible and wrist_visible:

    cv2.line(
        overlay,
        tuple(LE.astype(int)),
        tuple(LW.astype(int)),
        (255, 0, 0),
        2
    )


# ============================================================
# 23) แสดงผล
# ============================================================
cv2.imshow(
    "1 - Original",
    img
)

cv2.imshow(
    "2 - Person Mask",
    M_person_display
)

cv2.imshow(
    "3 - Torso Safety Mask",
    M_torso_display
)

cv2.imshow(
    "4 - Arm Final Before Clean",
    M_arm_final_display
)

cv2.imshow(
    "5 - Deform Mask",
    M_arm_clean_display
)

cv2.imshow(
    "6 - Blend Mask",
    M_arm_blend_display
)

cv2.imshow(
    "7 - Arm Overlay",
    overlay
)

cv2.imshow(
    "8 - Resize Result",
    result_resize
)


# ============================================================
# 24) ปิดด้วย ESC หรือ X
# ============================================================
while True:

    key = cv2.waitKey(100)

    if key == 27:
        break

    if cv2.getWindowProperty(
        "8 - Resize Result",
        cv2.WND_PROP_VISIBLE
    ) < 1:
        break


cv2.destroyAllWindows()
pose.close()