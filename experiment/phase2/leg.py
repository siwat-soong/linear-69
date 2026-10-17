"""
ไม่ยากถ้าอยากผอม : บีบ/ขยายขาซ้าย-ขวา
เวอร์ชันรองรับขางอ/ขาเอียงได้ดีขึ้น

แนวคิดหลัก
- แยกต้นขา (Hip -> Knee) และน่อง (Knee -> Ankle) เป็นคนละ deformation field
- MediaPipe ใช้หา Landmark + Person Segmentation
- ใช้ Arm Protection Mask กันมือ/แขนไม่ให้โดน deformation ของขา
- ใช้ Torso Protection กันลำตัว
- ใช้ Vector Projection + Euclidean Distance + Hadamard Product
- ใช้ Local Scaling + Inverse Mapping + Bilinear Interpolation
- แสดงหน้าต่าง debug หลายขั้นตอน

หน้าต่าง
1 - Original
2 - Person Mask
3 - Torso Protection
4 - Arm Protection
5 - Left Thigh Mask
6 - Left Calf Mask
7 - Right Thigh Mask
8 - Right Calf Mask
9 - Combined Leg Mask
10 - Leg Detection Overlay
11 - Final Result

กด ESC เพื่อปิดทุกหน้าต่าง
"""

import cv2
import numpy as np
import mediapipe as mp


# ============================================================
# 1. ตั้งค่า
# ============================================================

IMAGE_PATH = './dataset/img_0012_jpeg.jpg'

# -50 = ขาใหญ่ขึ้น
#   0 = ปกติ
# +50 = ขาเล็กลง
RESHAPE_LEVEL = 20

OUTPUT_PATH = './result_legs_segmented.jpg'

# ขนาด mask
THIGH_RADIUS = 32
CALF_RADIUS = 24
LEG_FEATHER = 28

# Arm Protection
ARM_UPPER_RADIUS = 24
ARM_LOWER_RADIUS = 20
HAND_EXTRA_RADIUS = 14

VISIBILITY_THRESHOLD = 0.65


# ============================================================
# 2. Pixel Grid
# ============================================================

def pixel_grid(h, w):
    X, Y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )

    return np.stack(
        [X, Y],
        axis=-1
    )


# ============================================================
# 3. Vector Projection + Euclidean Distance
# ============================================================

def closest_point_on_segment(P, A, B):
    """
    AB = B - A
    AP = P - A

    t = (AP·AB) / (AB·AB)
    Q = A + tAB
    D = ||P-Q||
    """

    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)

    AB = B - A
    AB_dot_AB = float(AB @ AB)

    if AB_dot_AB < 1e-6:
        Q = np.broadcast_to(A, P.shape).copy()
    else:
        AP = P - A

        t = (
            (AP @ AB)
            /
            AB_dot_AB
        )

        t = np.clip(
            t,
            0.0,
            1.0
        )

        Q = (
            A
            +
            t[..., None] * AB
        )

    D = np.linalg.norm(
        P - Q,
        axis=-1
    )

    return (
        Q.astype(np.float32),
        D.astype(np.float32)
    )


# ============================================================
# 4. Weight Function
# ============================================================

def weight_W(r, R):
    """
    W(r) = (1 - r^2/R^2)^2
    """

    q = np.clip(
        r / R,
        0.0,
        1.0
    )

    return (
        1.0 - q ** 2
    ) ** 2


# ============================================================
# 5. Field ของ segment เดียว
# ============================================================

def build_single_segment_field(
    h,
    w,
    A,
    B,
    radius,
    feather
):
    """
    สร้าง field สำหรับ segment เดียว
    เช่น Hip -> Knee หรือ Knee -> Ankle
    """

    P = pixel_grid(h, w)

    Q, D = closest_point_on_segment(
        P,
        A,
        B
    )

    r = np.maximum(
        D - radius,
        0.0
    )

    W = weight_W(
        r,
        feather
    )

    M_active = (
        D <= radius + feather
    ).astype(np.float32)

    G = (
        W
        *
        M_active
    )

    return (
        Q.astype(np.float32),
        G.astype(np.float32)
    )


# ============================================================
# 6. Arm Protection Mask
# ============================================================

def build_arm_protection_mask(
    h,
    w,
    shoulder,
    elbow,
    wrist,
    upper_ok,
    lower_ok
):
    P = pixel_grid(h, w)

    M_upper = np.zeros(
        (h, w),
        dtype=np.float32
    )

    M_lower = np.zeros(
        (h, w),
        dtype=np.float32
    )

    # Upper arm
    if upper_ok:

        _, D_upper = closest_point_on_segment(
            P,
            shoulder,
            elbow
        )

        M_upper = (
            D_upper
            <= ARM_UPPER_RADIUS
        ).astype(np.float32)

    # Lower arm
    if lower_ok:

        _, D_lower = closest_point_on_segment(
            P,
            elbow,
            wrist
        )

        M_lower = (
            D_lower
            <= ARM_LOWER_RADIUS
        ).astype(np.float32)

    M_arm = np.maximum(
        M_upper,
        M_lower
    )

    # เพิ่มวงกลม protect มือรอบ wrist
    if lower_ok:

        X = P[..., 0]
        Y = P[..., 1]

        D_hand = np.sqrt(
            (X - wrist[0]) ** 2
            +
            (Y - wrist[1]) ** 2
        )

        hand_radius = (
            ARM_LOWER_RADIUS
            +
            HAND_EXTRA_RADIUS
        )

        M_hand = (
            D_hand
            <= hand_radius
        ).astype(np.float32)

        M_arm = np.maximum(
            M_arm,
            M_hand
        )

    # ขยายขอบเล็กน้อย
    kernel = np.ones(
        (7, 7),
        dtype=np.uint8
    )

    M_arm = cv2.dilate(
        M_arm,
        kernel,
        iterations=1
    )

    M_arm = cv2.GaussianBlur(
        M_arm,
        (9, 9),
        0
    )

    return np.clip(
        M_arm,
        0.0,
        1.0
    )


# ============================================================
# 7. Bilinear Interpolation
# ============================================================

def bilinear_sample(
    img,
    map_x,
    map_y
):
    h, w = img.shape[:2]

    x = np.clip(
        map_x,
        0,
        w - 1.001
    )

    y = np.clip(
        map_y,
        0,
        h - 1.001
    )

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)

    x1 = x0 + 1
    y1 = y0 + 1

    wx = (
        x - x0
    )[..., None]

    wy = (
        y - y0
    )[..., None]

    f = img.astype(np.float32)

    top = (
        f[y0, x0] * (1.0 - wx)
        +
        f[y0, x1] * wx
    )

    bottom = (
        f[y1, x0] * (1.0 - wx)
        +
        f[y1, x1] * wx
    )

    result = (
        top * (1.0 - wy)
        +
        bottom * wy
    )

    return np.clip(
        result,
        0,
        255
    ).astype(np.uint8)


# ============================================================
# 8. Local Scaling + Inverse Mapping
# ============================================================

def reshape_legs(
    img,
    P,
    fields,
    scale
):
    """
    fields = [(Q1,G1), (Q2,G2), ...]

    แต่ละ field เป็นคนละ segment:
    left thigh
    left calf
    right thigh
    right calf
    """

    if abs(
        scale - 1.0
    ) < 1e-6:

        return img.copy()

    V_total = np.zeros_like(
        P,
        dtype=np.float32
    )

    for Q, G in fields:

        # local scale
        s = (
            1.0
            +
            (scale - 1.0) * G
        )

        s = np.maximum(
            s,
            0.05
        )

        # inverse mapping
        source = (
            Q
            +
            (P - Q)
            /
            s[..., None]
        )

        # displacement
        V = (
            source - P
        )

        V_total += V

    source_all = (
        P + V_total
    )

    return bilinear_sample(
        img,
        source_all[..., 0],
        source_all[..., 1]
    )


# ============================================================
# 9. Overlay Helper
# ============================================================

def make_overlay(
    img,
    mask,
    color='red'
):
    overlay = img.copy()

    layer = np.zeros_like(
        img
    )

    mask_u8 = (
        np.clip(
            mask,
            0.0,
            1.0
        )
        *
        255
    ).astype(np.uint8)

    if color == 'red':
        layer[:, :, 2] = mask_u8

    elif color == 'green':
        layer[:, :, 1] = mask_u8

    elif color == 'blue':
        layer[:, :, 0] = mask_u8

    elif color == 'yellow':
        layer[:, :, 1] = mask_u8
        layer[:, :, 2] = mask_u8

    elif color == 'cyan':
        layer[:, :, 0] = mask_u8
        layer[:, :, 1] = mask_u8

    return cv2.addWeighted(
        overlay,
        1.0,
        layer,
        0.5,
        0
    )


# ============================================================
# 10. โปรแกรมหลัก
# ============================================================

def main():

    # --------------------------------------------------------
    # อ่านภาพ
    # --------------------------------------------------------

    img = cv2.imread(
        IMAGE_PATH
    )

    if img is None:

        print(
            "ไม่พบไฟล์รูปภาพ:",
            IMAGE_PATH
        )

        return

    h, w = img.shape[:2]

    img_rgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )


    # ========================================================
    # MediaPipe Pose
    # ========================================================

    mp_pose = mp.solutions.pose

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=True
    ) as pose:

        result = pose.process(
            img_rgb
        )

    if (
        not result.pose_landmarks
        or
        result.segmentation_mask is None
    ):

        print(
            "ไม่พบ pose landmarks / person mask"
        )

        return

    lm = result.pose_landmarks.landmark
    L = mp_pose.PoseLandmark

    def pt(index):

        return np.array(
            [
                lm[index].x * w,
                lm[index].y * h
            ],
            dtype=np.float32
        )


    # ========================================================
    # 11. Person Mask
    # ========================================================

    M_person = (
        result.segmentation_mask
        >= 0.30
    ).astype(np.float32)

    k = (
        2 * LEG_FEATHER
        +
        1
    )

    M_person = cv2.dilate(
        M_person,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (k, k)
        )
    )

    M_person = cv2.GaussianBlur(
        M_person,
        (0, 0),
        LEG_FEATHER / 3
    )

    M_person = np.clip(
        M_person,
        0.0,
        1.0
    )


    # ========================================================
    # 12. Torso Protection
    # ========================================================

    torso_points = np.array(
        [
            pt(L.LEFT_SHOULDER),
            pt(L.RIGHT_SHOULDER),
            pt(L.RIGHT_HIP),
            pt(L.LEFT_HIP)
        ],
        dtype=np.int32
    )

    M_torso = np.zeros(
        (h, w),
        dtype=np.float32
    )

    cv2.fillPoly(
        M_torso,
        [torso_points],
        1.0
    )

    M_torso = cv2.dilate(
        M_torso,
        np.ones(
            (7, 7),
            dtype=np.uint8
        ),
        iterations=1
    )

    M_torso = cv2.GaussianBlur(
        M_torso,
        (7, 7),
        0
    )

    M_torso = np.clip(
        M_torso,
        0.0,
        1.0
    )


    # ========================================================
    # 13. Arm Protection
    # ========================================================

    arm_definitions = {
        'left': (
            L.LEFT_SHOULDER,
            L.LEFT_ELBOW,
            L.LEFT_WRIST
        ),

        'right': (
            L.RIGHT_SHOULDER,
            L.RIGHT_ELBOW,
            L.RIGHT_WRIST
        )
    }

    M_arm_protect = np.zeros(
        (h, w),
        dtype=np.float32
    )

    for name, (
        shoulder_id,
        elbow_id,
        wrist_id
    ) in arm_definitions.items():

        shoulder_vis = lm[
            shoulder_id
        ].visibility

        elbow_vis = lm[
            elbow_id
        ].visibility

        wrist_vis = lm[
            wrist_id
        ].visibility

        upper_ok = (
            shoulder_vis
            >= VISIBILITY_THRESHOLD
            and
            elbow_vis
            >= VISIBILITY_THRESHOLD
        )

        lower_ok = (
            elbow_vis
            >= VISIBILITY_THRESHOLD
            and
            wrist_vis
            >= VISIBILITY_THRESHOLD
        )

        print()
        print(
            f"{name.upper()} ARM PROTECTION"
        )

        print(
            "visibility:",
            [
                round(
                    shoulder_vis,
                    3
                ),
                round(
                    elbow_vis,
                    3
                ),
                round(
                    wrist_vis,
                    3
                )
            ]
        )

        print(
            "upper:",
            "DETECT"
            if upper_ok
            else
            "SKIP"
        )

        print(
            "lower:",
            "DETECT"
            if lower_ok
            else
            "SKIP"
        )

        if not (
            upper_ok
            or
            lower_ok
        ):
            continue

        shoulder = pt(
            shoulder_id
        )

        elbow = pt(
            elbow_id
        )

        wrist = pt(
            wrist_id
        )

        M_one_arm = build_arm_protection_mask(
            h,
            w,
            shoulder,
            elbow,
            wrist,
            upper_ok,
            lower_ok
        )

        M_arm_protect = np.maximum(
            M_arm_protect,
            M_one_arm
        )

    M_arm_protect = np.clip(
        M_arm_protect,
        0.0,
        1.0
    )


    # ========================================================
    # 14. Leg Definitions
    # ========================================================

    legs = {
        'left': (
            L.LEFT_HIP,
            L.LEFT_KNEE,
            L.LEFT_ANKLE
        ),

        'right': (
            L.RIGHT_HIP,
            L.RIGHT_KNEE,
            L.RIGHT_ANKLE
        )
    }

    P = pixel_grid(
        h,
        w
    )

    fields = []

    masks = {
        'left_thigh': np.zeros(
            (h, w),
            dtype=np.float32
        ),

        'left_calf': np.zeros(
            (h, w),
            dtype=np.float32
        ),

        'right_thigh': np.zeros(
            (h, w),
            dtype=np.float32
        ),

        'right_calf': np.zeros(
            (h, w),
            dtype=np.float32
        )
    }

    detection_info = []


    # ========================================================
    # 15. Detect + Field แยกต้นขา/น่อง
    # ========================================================

    for name, (
        hip_id,
        knee_id,
        ankle_id
    ) in legs.items():

        hip_visibility = lm[
            hip_id
        ].visibility

        knee_visibility = lm[
            knee_id
        ].visibility

        ankle_visibility = lm[
            ankle_id
        ].visibility

        thigh_ok = (
            hip_visibility
            >= VISIBILITY_THRESHOLD
            and
            knee_visibility
            >= VISIBILITY_THRESHOLD
        )

        calf_ok = (
            knee_visibility
            >= VISIBILITY_THRESHOLD
            and
            ankle_visibility
            >= VISIBILITY_THRESHOLD
        )

        print()
        print(
            "=============================="
        )

        print(
            f"{name.upper()} LEG"
        )

        print(
            "=============================="
        )

        print(
            "Hip visibility   =",
            round(
                hip_visibility,
                3
            )
        )

        print(
            "Knee visibility  =",
            round(
                knee_visibility,
                3
            )
        )

        print(
            "Ankle visibility =",
            round(
                ankle_visibility,
                3
            )
        )

        print(
            "Thigh:",
            "DETECT"
            if thigh_ok
            else
            "SKIP"
        )

        print(
            "Calf:",
            "DETECT"
            if calf_ok
            else
            "SKIP"
        )

        if not (
            thigh_ok
            or
            calf_ok
        ):
            continue

        H = pt(
            hip_id
        )

        K = pt(
            knee_id
        )

        A = pt(
            ankle_id
        )

        # ----------------------------------------------------
        # Thigh Field
        # ----------------------------------------------------

        if thigh_ok:

            Q_thigh, G_thigh = build_single_segment_field(
                h,
                w,
                H,
                K,
                THIGH_RADIUS,
                LEG_FEATHER
            )

            # Hadamard Product
            G_thigh = (
                G_thigh
                *
                M_person
                *
                (1.0 - M_torso)
                *
                (1.0 - M_arm_protect)
            )

            # soften ขอบ
            G_thigh = cv2.GaussianBlur(
                G_thigh,
                (7, 7),
                0
            )

            G_thigh = np.clip(
                G_thigh,
                0.0,
                1.0
            )

            fields.append(
                (
                    Q_thigh,
                    G_thigh
                )
            )

            masks[
                f'{name}_thigh'
            ] = G_thigh


        # ----------------------------------------------------
        # Calf Field
        # ----------------------------------------------------

        if calf_ok:

            Q_calf, G_calf = build_single_segment_field(
                h,
                w,
                K,
                A,
                CALF_RADIUS,
                LEG_FEATHER
            )

            G_calf = (
                G_calf
                *
                M_person
                *
                (1.0 - M_torso)
                *
                (1.0 - M_arm_protect)
            )

            G_calf = cv2.GaussianBlur(
                G_calf,
                (7, 7),
                0
            )

            G_calf = np.clip(
                G_calf,
                0.0,
                1.0
            )

            fields.append(
                (
                    Q_calf,
                    G_calf
                )
            )

            masks[
                f'{name}_calf'
            ] = G_calf


        detection_info.append(
            (
                name,
                H,
                K,
                A,
                thigh_ok,
                calf_ok
            )
        )


    # ========================================================
    # 16. เช็ก Field
    # ========================================================

    if not fields:

        print(
            "ไม่พบขาที่สามารถ reshape ได้"
        )

        return


    # ========================================================
    # 17. Combined Leg Mask
    # ========================================================

    M_leg_all = np.maximum.reduce(
        [
            masks[
                'left_thigh'
            ],
            masks[
                'left_calf'
            ],
            masks[
                'right_thigh'
            ],
            masks[
                'right_calf'
            ]
        ]
    )


    # ========================================================
    # 18. RESHAPE_LEVEL -> scale
    # ========================================================

    level = int(
        np.clip(
            RESHAPE_LEVEL,
            -50,
            50
        )
    )

    scale = (
        1.0
        -
        level / 100.0
    )

    print()
    print(
        "=============================="
    )

    print(
        "RESHAPE"
    )

    print(
        "=============================="
    )

    print(
        "RESHAPE_LEVEL =",
        level
    )

    print(
        "scale =",
        round(
            scale,
            3
        )
    )


    # ========================================================
    # 19. Reshape
    # ========================================================

    result_img = reshape_legs(
        img,
        P,
        fields,
        scale
    )


    # ========================================================
    # 20. Debug Images
    # ========================================================

    person_display = (
        M_person * 255
    ).astype(np.uint8)

    torso_display = (
        M_torso * 255
    ).astype(np.uint8)

    arm_protect_overlay = make_overlay(
        img,
        M_arm_protect,
        'yellow'
    )

    left_thigh_overlay = make_overlay(
        img,
        masks[
            'left_thigh'
        ],
        'red'
    )

    left_calf_overlay = make_overlay(
        img,
        masks[
            'left_calf'
        ],
        'green'
    )

    right_thigh_overlay = make_overlay(
        img,
        masks[
            'right_thigh'
        ],
        'blue'
    )

    right_calf_overlay = make_overlay(
        img,
        masks[
            'right_calf'
        ],
        'cyan'
    )

    combined_leg_overlay = make_overlay(
        img,
        M_leg_all,
        'red'
    )


    # ========================================================
    # 21. Detection Overlay
    # ========================================================

    detection_overlay = img.copy()

    for (
        name,
        H,
        K,
        A,
        thigh_ok,
        calf_ok
    ) in detection_info:

        if name == 'left':

            thigh_color = (
                255,
                0,
                0
            )

            calf_color = (
                0,
                255,
                0
            )

        else:

            thigh_color = (
                0,
                0,
                255
            )

            calf_color = (
                255,
                255,
                0
            )

        # joints
        for p in (
            H,
            K,
            A
        ):

            cv2.circle(
                detection_overlay,
                tuple(
                    p.astype(int)
                ),
                7,
                (
                    0,
                    255,
                    255
                ),
                -1
            )

        if thigh_ok:

            cv2.line(
                detection_overlay,
                tuple(
                    H.astype(int)
                ),
                tuple(
                    K.astype(int)
                ),
                thigh_color,
                3
            )

        if calf_ok:

            cv2.line(
                detection_overlay,
                tuple(
                    K.astype(int)
                ),
                tuple(
                    A.astype(int)
                ),
                calf_color,
                3
            )

        cv2.putText(
            detection_overlay,
            f"{name} leg",
            tuple(
                (
                    K
                    +
                    np.array(
                        [10, 0]
                    )
                ).astype(int)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (
                0,
                255,
                255
            ),
            2
        )


    # ========================================================
    # 22. Save
    # ========================================================

    cv2.imwrite(
        OUTPUT_PATH,
        result_img
    )

    print()
    print(
        "Saved:",
        OUTPUT_PATH
    )


    # ========================================================
    # 23. Show Windows
    # ========================================================

    cv2.imshow(
        "1 - Original",
        img
    )

    cv2.imshow(
        "2 - Person Mask",
        person_display
    )

    cv2.imshow(
        "3 - Torso Protection",
        torso_display
    )

    cv2.imshow(
        "4 - Arm Protection",
        arm_protect_overlay
    )

    cv2.imshow(
        "5 - Left Thigh Mask",
        left_thigh_overlay
    )

    cv2.imshow(
        "6 - Left Calf Mask",
        left_calf_overlay
    )

    cv2.imshow(
        "7 - Right Thigh Mask",
        right_thigh_overlay
    )

    cv2.imshow(
        "8 - Right Calf Mask",
        right_calf_overlay
    )

    cv2.imshow(
        "9 - Combined Leg Mask",
        combined_leg_overlay
    )

    cv2.imshow(
        "10 - Leg Detection Overlay",
        detection_overlay
    )

    cv2.imshow(
        "11 - Final Result",
        result_img
    )


    # ========================================================
    # 24. Wait
    # ========================================================

    print(
        "กด ESC เพื่อปิดหน้าต่าง"
    )

    while True:

        key = (
            cv2.waitKey(100)
            &
            0xFF
        )

        if key == 27:
            break

        if cv2.getWindowProperty(
            "11 - Final Result",
            cv2.WND_PROP_VISIBLE
        ) < 1:
            break

    cv2.destroyAllWindows()


# ============================================================
# 25. Run
# ============================================================

if __name__ == "__main__":
    main()
