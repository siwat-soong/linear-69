import cv2
import numpy as np
import mediapipe as mp


# ============================================================
# ฟังก์ชันหาระยะจากทุก pixel ไปยังเส้นช่วง AB
# ============================================================

def point_to_segment_distance_map(height, width, A, B):
    Ax, Ay = A
    Bx, By = B

    X, Y = np.meshgrid(
        np.arange(width),
        np.arange(height)
    )

    P = np.stack(
        [X, Y],
        axis=-1
    ).astype(np.float32)

    A = np.array(
        [Ax, Ay],
        dtype=np.float32
    )

    B = np.array(
        [Bx, By],
        dtype=np.float32
    )

    AB = B - A
    AB_len_sq = np.sum(AB ** 2)

    # กันกรณี A และ B ซ้อนกัน
    if AB_len_sq < 1e-6:
        return np.sqrt(
            (X - Ax) ** 2 +
            (Y - Ay) ** 2
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

    # จุดที่ใกล้ P ที่สุดบนเส้น AB
    projection = (
        A +
        t[..., None] * AB
    )

    diff = P - projection

    # Euclidean Distance
    dist = np.sqrt(
        np.sum(
            diff ** 2,
            axis=-1
        )
    )

    return dist


# ============================================================
# สร้าง Soft Capsule Mask
# ============================================================

def capsule_soft_mask(
    height,
    width,
    A,
    B,
    radius,
    sigma=8.0
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
            (2 * sigma ** 2)
        )
    )

    return mask.astype(np.float32)


# ============================================================
# 1. อ่านภาพ
# ============================================================

img = cv2.imread(
    './dataset/img_0190_jpeg.jpg'
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
# 2. MediaPipe Pose
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
# 3. สร้าง M_person
# ============================================================

if result.segmentation_mask is None:
    print("ไม่พบ Person Segmentation Mask")
    pose.close()
    exit()

M_person = result.segmentation_mask.astype(
    np.float32
)

# ตัดค่าที่ confidence ต่ำมาก
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
# 4. สร้าง M_torso
# ใช้ Shoulder + Hip เป็น polygon
# ============================================================

right_shoulder_torso = landmarks[
    mp_pose.PoseLandmark.RIGHT_SHOULDER
]

right_shoulder_torso = landmarks[
    mp_pose.PoseLandmark.RIGHT_SHOULDER
]

right_hip = landmarks[
    mp_pose.PoseLandmark.RIGHT_HIP
]

right_hip = landmarks[
    mp_pose.PoseLandmark.RIGHT_HIP
]


LS_torso = np.array(
    [
        right_shoulder_torso.x * width,
        right_shoulder_torso.y * height
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
        right_hip.x * width,
        right_hip.y * height
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

# เกลี่ยขอบ torso mask
M_torso = cv2.GaussianBlur(
    M_torso,
    (11, 11),
    0
)

M_torso = np.clip(
    M_torso,
    0.0,
    1.0
)


# ============================================================
# 5. ดึง Landmark แขนซ้าย
# ============================================================

right_shoulder = landmarks[
    mp_pose.PoseLandmark.RIGHT_SHOULDER
]

right_elbow = landmarks[
    mp_pose.PoseLandmark.RIGHT_ELBOW
]

right_wrist = landmarks[
    mp_pose.PoseLandmark.RIGHT_WRIST
]


# normalized coordinate -> pixel coordinate

LS = np.array(
    [
        right_shoulder.x * width,
        right_shoulder.y * height
    ],
    dtype=np.float32
)

LE = np.array(
    [
        right_elbow.x * width,
        right_elbow.y * height
    ],
    dtype=np.float32
)

LW = np.array(
    [
        right_wrist.x * width,
        right_wrist.y * height
    ],
    dtype=np.float32
)


# ============================================================
# 6. Visibility Threshold
# ============================================================

visibility_threshold = 0.65

shoulder_visible = (
    right_shoulder.visibility >= visibility_threshold
)

elbow_visible = (
    right_elbow.visibility >= visibility_threshold
)

wrist_visible = (
    right_wrist.visibility >= visibility_threshold
)


print(
    "Right shoulder visibility:",
    right_shoulder.visibility
)

print(
    "Right elbow visibility:",
    right_elbow.visibility
)

print(
    "Right wrist visibility:",
    right_wrist.visibility
)


# ============================================================
# 7. กำหนดความหนาแขน
# ============================================================

upper_radius = 22
lower_radius = 18


# ============================================================
# 8. เตรียม Zero Mask
# ============================================================

M_upper = np.zeros(
    (height, width),
    dtype=np.float32
)

M_lower = np.zeros(
    (height, width),
    dtype=np.float32
)


# ============================================================
# 9. ต้นแขน
# Shoulder -> Elbow
# ============================================================

if shoulder_visible and elbow_visible:

    M_upper = capsule_soft_mask(
        height,
        width,
        LS,
        LE,
        upper_radius,
        sigma=10.0
    )

    print("Detect: Right upper arm")

else:

    print("Skip: Right upper arm")


# ============================================================
# 10. ปลายแขน
# Elbow -> Wrist
# ============================================================

if elbow_visible and wrist_visible:

    M_lower = capsule_soft_mask(
        height,
        width,
        LE,
        LW,
        lower_radius,
        sigma=10.0
    )

    print("Detect: Right lower arm")

else:

    print("Skip: Right lower arm")


# ============================================================
# 11. รวมต้นแขน + ปลายแขน
# ============================================================

M_arm = np.maximum(
    M_upper,
    M_lower
)


# ============================================================
# 12. เกลี่ยขอบ Arm Mask
# ============================================================

M_arm_blur = cv2.GaussianBlur(
    M_arm,
    (11, 11),
    0
)


# ============================================================
# 13. Final Mask
#
# Arm Capsule
# × Person Mask
# × ไม่เอา Torso
#
# M_arm_final =
# M_arm × M_person × (1 - M_torso)
# ============================================================

M_arm_final = (
    M_arm_blur
    * M_person
    * (1.0 - M_torso)
)

M_arm_final = np.clip(
    M_arm_final,
    0.0,
    1.0
)


# ============================================================
# 14. เตรียมภาพสำหรับแสดงผล
# ============================================================

M_person_display = (
    M_person * 255
).astype(np.uint8)

M_torso_display = (
    M_torso * 255
).astype(np.uint8)

M_arm_before_display = (
    M_arm_blur * 255
).astype(np.uint8)

M_arm_final_display = (
    M_arm_final * 255
).astype(np.uint8)


# ============================================================
# 15. Overlay ลงบนภาพจริง
# ============================================================

overlay = img.copy()

red_layer = np.zeros_like(
    img
)

red_layer[:, :, 2] = (
    M_arm_final_display
)

overlay = cv2.addWeighted(
    overlay,
    1.0,
    red_layer,
    0.5,
    0
)


# ============================================================
# 16. วาด Landmark
# ============================================================

# วาดเฉพาะ landmark ที่ visibility ผ่าน
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


# วาดต้นแขนถ้าตรวจพบ
if shoulder_visible and elbow_visible:

    cv2.line(
        overlay,
        tuple(LS.astype(int)),
        tuple(LE.astype(int)),
        (255, 0, 0),
        2
    )


# วาดปลายแขนถ้าตรวจพบ
if elbow_visible and wrist_visible:

    cv2.line(
        overlay,
        tuple(LE.astype(int)),
        tuple(LW.astype(int)),
        (255, 0, 0),
        2
    )


# ============================================================
# 17. แสดงผล
# ============================================================

cv2.imshow(
    "1 - Person Mask",
    M_person_display
)

cv2.imshow(
    "2 - Torso Mask",
    M_torso_display
)

cv2.imshow(
    "3 - Arm Before Person/Torso",
    M_arm_before_display
)

cv2.imshow(
    "4 - Right Arm Final",
    M_arm_final_display
)

cv2.imshow(
    "5 - Overlay",
    overlay
)


# ============================================================
# 18. ปิดหน้าต่างแบบกด ESC หรือ X
# ============================================================

while True:

    key = cv2.waitKey(100)

    # กด ESC
    if key == 27:
        break

    # กด X ที่หน้าต่าง Overlay
    if cv2.getWindowProperty(
        "5 - Overlay",
        cv2.WND_PROP_VISIBLE
    ) < 1:
        break


cv2.destroyAllWindows()
pose.close()