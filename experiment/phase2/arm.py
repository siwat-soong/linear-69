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

    A = np.array(A, dtype=np.float32)
    B = np.array(B, dtype=np.float32)

    AB = B - A

    AB_len_sq = np.sum(
        AB ** 2
    )

    # กันกรณี A กับ B ซ้อนกัน
    if AB_len_sq < 1e-6:

        return np.sqrt(
            (X - Ax) ** 2 +
            (Y - Ay) ** 2
        )

    AP = P - A

    # Projection parameter
    #
    #       AP · AB
    # t =  ---------
    #       AB · AB
    #
    t = np.sum(
        AP * AB,
        axis=-1
    ) / AB_len_sq

    # จำกัดให้อยู่บน Segment AB
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

    return mask.astype(
        np.float32
    )


# ============================================================
# 1. อ่านภาพ
# ============================================================

img = cv2.imread(
    './dataset/img_0214_jpeg.jpg'
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
# 3. Person Segmentation Mask
# ============================================================

if result.segmentation_mask is None:

    print("ไม่พบ Person Segmentation Mask")

    pose.close()

    exit()


M_person = (
    result.segmentation_mask
    .astype(np.float32)
)


# threshold
M_person = np.where(
    M_person >= 0.30,
    M_person,
    0.0
).astype(np.float32)


# smooth
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
# 4. Torso Exclusion Mask
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
# 5. Landmark แขนซ้าย
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
# 6. Landmark แขนขวา
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


RS = np.array(
    [
        right_shoulder.x * width,
        right_shoulder.y * height
    ],
    dtype=np.float32
)


RE = np.array(
    [
        right_elbow.x * width,
        right_elbow.y * height
    ],
    dtype=np.float32
)


RW = np.array(
    [
        right_wrist.x * width,
        right_wrist.y * height
    ],
    dtype=np.float32
)


# ============================================================
# 7. Visibility
# ============================================================

visibility_threshold = 0.65


# ---------- LEFT ----------

LS_visible = (
    left_shoulder.visibility
    >= visibility_threshold
)

LE_visible = (
    left_elbow.visibility
    >= visibility_threshold
)

LW_visible = (
    left_wrist.visibility
    >= visibility_threshold
)


# ---------- RIGHT ----------

RS_visible = (
    right_shoulder.visibility
    >= visibility_threshold
)

RE_visible = (
    right_elbow.visibility
    >= visibility_threshold
)

RW_visible = (
    right_wrist.visibility
    >= visibility_threshold
)


print("\n========== LEFT ARM ==========")

print(
    "Shoulder:",
    round(left_shoulder.visibility, 3)
)

print(
    "Elbow:",
    round(left_elbow.visibility, 3)
)

print(
    "Wrist:",
    round(left_wrist.visibility, 3)
)


print("\n========== RIGHT ARM ==========")

print(
    "Shoulder:",
    round(right_shoulder.visibility, 3)
)

print(
    "Elbow:",
    round(right_elbow.visibility, 3)
)

print(
    "Wrist:",
    round(right_wrist.visibility, 3)
)


# ============================================================
# 8. ความหนาแขน
# ============================================================

upper_radius = 22
lower_radius = 18


# ============================================================
# 9. Zero Mask
# ============================================================

M_left_upper = np.zeros(
    (height, width),
    dtype=np.float32
)

M_left_lower = np.zeros(
    (height, width),
    dtype=np.float32
)


M_right_upper = np.zeros(
    (height, width),
    dtype=np.float32
)

M_right_lower = np.zeros(
    (height, width),
    dtype=np.float32
)


# ============================================================
# 10. LEFT UPPER ARM
# Shoulder -> Elbow
# ============================================================

if LS_visible and LE_visible:

    M_left_upper = capsule_soft_mask(
        height,
        width,
        LS,
        LE,
        upper_radius,
        sigma=10.0
    )

    print(
        "\nDetect: Left upper arm"
    )

else:

    print(
        "\nSkip: Left upper arm"
    )


# ============================================================
# 11. LEFT LOWER ARM
# Elbow -> Wrist
# ============================================================

if LE_visible and LW_visible:

    M_left_lower = capsule_soft_mask(
        height,
        width,
        LE,
        LW,
        lower_radius,
        sigma=10.0
    )

    print(
        "Detect: Left lower arm"
    )

else:

    print(
        "Skip: Left lower arm"
    )


# ============================================================
# 12. RIGHT UPPER ARM
# Shoulder -> Elbow
# ============================================================

if RS_visible and RE_visible:

    M_right_upper = capsule_soft_mask(
        height,
        width,
        RS,
        RE,
        upper_radius,
        sigma=10.0
    )

    print(
        "Detect: Right upper arm"
    )

else:

    print(
        "Skip: Right upper arm"
    )


# ============================================================
# 13. RIGHT LOWER ARM
# Elbow -> Wrist
# ============================================================

if RE_visible and RW_visible:

    M_right_lower = capsule_soft_mask(
        height,
        width,
        RE,
        RW,
        lower_radius,
        sigma=10.0
    )

    print(
        "Detect: Right lower arm"
    )

else:

    print(
        "Skip: Right lower arm"
    )


# ============================================================
# 14. รวมแขนแต่ละข้าง
# ============================================================

M_left_arm = np.maximum(
    M_left_upper,
    M_left_lower
)


M_right_arm = np.maximum(
    M_right_upper,
    M_right_lower
)


# ============================================================
# 15. Smooth
# ============================================================

M_left_arm = cv2.GaussianBlur(
    M_left_arm,
    (11, 11),
    0
)


M_right_arm = cv2.GaussianBlur(
    M_right_arm,
    (11, 11),
    0
)


# ============================================================
# 16. Person Mask + Torso Exclusion
# ============================================================

M_left_arm_final = (
    M_left_arm
    * M_person
    * (1.0 - M_torso)
)


M_right_arm_final = (
    M_right_arm
    * M_person
    * (1.0 - M_torso)
)


M_left_arm_final = np.clip(
    M_left_arm_final,
    0.0,
    1.0
)


M_right_arm_final = np.clip(
    M_right_arm_final,
    0.0,
    1.0
)


# ============================================================
# 17. รวมแขนซ้าย + แขนขวา
# ============================================================

M_arms = np.maximum(
    M_left_arm_final,
    M_right_arm_final
)


# ============================================================
# 18. Convert สำหรับแสดงผล
# ============================================================

left_display = (
    M_left_arm_final * 255
).astype(np.uint8)


right_display = (
    M_right_arm_final * 255
).astype(np.uint8)


arms_display = (
    M_arms * 255
).astype(np.uint8)


person_display = (
    M_person * 255
).astype(np.uint8)


torso_display = (
    M_torso * 255
).astype(np.uint8)


# ============================================================
# 19. Overlay
# ============================================================

overlay = img.copy()


red_layer = np.zeros_like(
    img
)


red_layer[:, :, 2] = (
    arms_display
)


overlay = cv2.addWeighted(
    overlay,
    1.0,
    red_layer,
    0.5,
    0
)


# ============================================================
# 20. วาด Landmark
# ============================================================

# LEFT

if LS_visible:

    cv2.circle(
        overlay,
        tuple(LS.astype(int)),
        6,
        (0, 255, 0),
        -1
    )


if LE_visible:

    cv2.circle(
        overlay,
        tuple(LE.astype(int)),
        6,
        (0, 255, 0),
        -1
    )


if LW_visible:

    cv2.circle(
        overlay,
        tuple(LW.astype(int)),
        6,
        (0, 255, 0),
        -1
    )


if LS_visible and LE_visible:

    cv2.line(
        overlay,
        tuple(LS.astype(int)),
        tuple(LE.astype(int)),
        (255, 0, 0),
        2
    )


if LE_visible and LW_visible:

    cv2.line(
        overlay,
        tuple(LE.astype(int)),
        tuple(LW.astype(int)),
        (255, 0, 0),
        2
    )


# RIGHT

if RS_visible:

    cv2.circle(
        overlay,
        tuple(RS.astype(int)),
        6,
        (0, 255, 0),
        -1
    )


if RE_visible:

    cv2.circle(
        overlay,
        tuple(RE.astype(int)),
        6,
        (0, 255, 0),
        -1
    )


if RW_visible:

    cv2.circle(
        overlay,
        tuple(RW.astype(int)),
        6,
        (0, 255, 0),
        -1
    )


if RS_visible and RE_visible:

    cv2.line(
        overlay,
        tuple(RS.astype(int)),
        tuple(RE.astype(int)),
        (255, 0, 0),
        2
    )


if RE_visible and RW_visible:

    cv2.line(
        overlay,
        tuple(RE.astype(int)),
        tuple(RW.astype(int)),
        (255, 0, 0),
        2
    )


# ============================================================
# 21. แสดงผล
# ============================================================

cv2.imshow(
    "1 - Person Mask",
    person_display
)

cv2.imshow(
    "2 - Torso Mask",
    torso_display
)

cv2.imshow(
    "3 - Left Arm",
    left_display
)

cv2.imshow(
    "4 - Right Arm",
    right_display
)

cv2.imshow(
    "5 - Both Arms",
    arms_display
)

cv2.imshow(
    "6 - Overlay",
    overlay
)


# ============================================================
# 22. ปิดด้วย ESC หรือ X
# ============================================================

while True:

    key = cv2.waitKey(100)

    if key == 27:
        break

    if cv2.getWindowProperty(
        "6 - Overlay",
        cv2.WND_PROP_VISIBLE
    ) < 1:
        break


cv2.destroyAllWindows()

pose.close()