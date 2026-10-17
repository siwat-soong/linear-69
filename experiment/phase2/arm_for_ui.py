"""
ไม่ยากถ้าอยากผอม : บีบ/ขยายแขนด้วย Vector-Matrix
เวอร์ชันอธิบาย Linear Algebra สำหรับส่งงาน

- ไม่มี UI / Trackbar
- MediaPipe ใช้หา landmark และ person mask เท่านั้น
- ใช้ Vector Projection, Dot Product, Euclidean Norm, Projection Matrix,
  Hadamard Product, Local Scale Matrix, Matrix-Vector Multiplication,
  Vector Addition และ Bilinear Interpolation
- แสดงการคำนวณใน Terminal และบันทึกลง linear_calculation.txt
"""

import cv2
import numpy as np
import mediapipe as mp

IMAGE_PATH = './dataset/img_0283_jpeg.jpg'
RESHAPE_LEVEL = 30  # -50 ใหญ่ขึ้น, 0 ปกติ, +50 เล็กลง
OUTPUT_PATH = './result_arms_linear.jpg'
CALCULATION_REPORT_PATH = './linear_calculation.txt'

UPPER_RADIUS = 22
LOWER_RADIUS = 18
FEATHER = 25
VISIBILITY_THRESHOLD = 0.65

CALC_LINES = []


def log(text=''):
    text = str(text)
    CALC_LINES.append(text)
    print(text)


def fmt_vec(v):
    v = np.asarray(v).reshape(-1)
    return '[' + ', '.join(f'{x:.4f}' for x in v) + ']'


def fmt_mat(M):
    return np.array2string(np.asarray(M), precision=4, suppress_small=True)


def pixel_grid(h, w):
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32),
                       np.arange(h, dtype=np.float32))
    return np.stack([X, Y], axis=-1)


def closest_point_on_segment(P, A, B):
    """
    AB = B-A
    AP = P-A
    t  = (AP·AB)/(AB·AB)
    Q  = A+tAB
    D  = ||P-Q||_2
    """
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
        'G': G.astype(np.float32),
        'D': D.astype(np.float32),
        'R': R.astype(np.float32),
        'omega': omega.astype(np.float32),
    }


def bilinear_sample(img, map_x, map_y):
    h, w = img.shape[:2]
    x = np.clip(map_x, 0, w - 1.001)
    y = np.clip(map_y, 0, h - 1.001)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    wx = (x - x0)[..., None]
    wy = (y - y0)[..., None]
    f = img.astype(np.float32)

    top = f[y0, x0] * (1.0 - wx) + f[y0, x1] * wx
    bottom = f[y1, x0] * (1.0 - wx) + f[y1, x1] * wx
    out = top * (1.0 - wy) + bottom * wy

    return np.clip(out, 0, 255).astype(np.uint8)


def report_segment(name, A, B):
    AB = B - A
    length = np.linalg.norm(AB)
    log()
    log(f'--- {name} ---')
    log(f'A = {fmt_vec(A)}')
    log(f'B = {fmt_vec(B)}')
    log(f'AB = B-A = {fmt_vec(AB)}')
    log(f'AB·AB = {float(AB @ AB):.4f}')
    log(f'||AB|| = {length:.4f}')

    if length < 1e-6:
        return

    u = AB / length
    P_parallel = np.outer(u, u)
    P_perp = np.eye(2, dtype=np.float32) - P_parallel

    log(f'u = AB/||AB|| = {fmt_vec(u)}')
    log('P_parallel = u u^T =')
    log(fmt_mat(P_parallel))
    log('P_perp = I - u u^T =')
    log(fmt_mat(P_perp))


def report_projection_example(name, A, B, radius):
    AB = B - A
    length = np.linalg.norm(AB)
    if length < 1e-6:
        return

    u = AB / length
    n = np.array([-u[1], u[0]], dtype=np.float32)
    P = (A + B) / 2.0 + n * (0.70 * radius)
    AP = P - A

    numerator = float(AP @ AB)
    denominator = float(AB @ AB)
    t_raw = numerator / denominator
    t = float(np.clip(t_raw, 0.0, 1.0))
    Q = A + t * AB
    PQ = P - Q
    D = float(np.linalg.norm(PQ))

    log()
    log(f'ตัวอย่าง Projection : {name}')
    log(f'P = {fmt_vec(P)}')
    log(f'AP = P-A = {fmt_vec(AP)}')
    log(f'AP·AB = {numerator:.4f}')
    log(f'AB·AB = {denominator:.4f}')
    log(f't = (AP·AB)/(AB·AB) = {t_raw:.4f}')
    log(f'Q = A+tAB = {fmt_vec(Q)}')
    log(f'P-Q = {fmt_vec(PQ)}')
    log(f'D = ||P-Q|| = {D:.4f} px')


def reshape_arms(img, P, fields, scale):
    """
    s(P) = 1 + (scale-1)G(P)

    S_inv(P) = [[1/s(P), 0],
                [0, 1/s(P)]]

    delta_source = S_inv(P) @ (P-Q)
    source       = Q + delta_source
    V_total      = sum(source-P)
    P_source     = P + V_total
    """
    if abs(scale - 1.0) < 1e-6:
        return img.copy()

    V_total = np.zeros_like(P, dtype=np.float32)

    for field in fields:
        Q = field['Q']
        G = field['G']
        s = np.maximum(1.0 + (scale - 1.0) * G, 0.05).astype(np.float32)

        # matrix field S_inv(P)
        S_inv = np.zeros(P.shape[:2] + (2, 2), dtype=np.float32)
        inv_s = 1.0 / s
        S_inv[..., 0, 0] = inv_s
        S_inv[..., 1, 1] = inv_s

        delta = P - Q
        delta_source = np.einsum('...ij,...j->...i', S_inv, delta)
        source = Q + delta_source
        V_total += source - P

    source_all = P + V_total
    return bilinear_sample(img, source_all[..., 0], source_all[..., 1])


def report_scale(level, scale):
    log()
    log('=' * 60)
    log('SCALE CALCULATION')
    log('=' * 60)
    log('scale = 1 - RESHAPE_LEVEL/100')
    log(f'      = 1 - ({level})/100')
    log(f'      = {scale:.4f}')

    local_s = scale
    inv_s = 1.0 / local_s
    S_inv = np.array([[inv_s, 0.0], [0.0, inv_s]], dtype=np.float32)

    log('เมื่อ G(P)=1:')
    log(f's(P) = 1 + (scale-1)G(P) = {local_s:.4f}')
    log('S_inv =')
    log(fmt_mat(S_inv))


def process_arm(image_path=IMAGE_PATH, reshape_level=RESHAPE_LEVEL):
    """
    ฟังก์ชันสำหรับเชื่อม UI / Backend

    Return:
        status = 1 -> สร้าง Arm Mask ได้
        status = 0 -> สร้าง Arm Mask ไม่ได้

        result = ภาพผลลัพธ์ หรือ None
        info   = dict รายละเอียดเพิ่มเติม
    """

    CALC_LINES.clear()

    img = cv2.imread(image_path)

    if img is None:
        print('ไม่พบไฟล์รูปภาพ:', image_path)
        return 0, None, {
            'status': 0,
            'message': 'image_not_found',
            'left_arm': 0,
            'right_arm': 0
        }

    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    log('=' * 60)
    log('LINEAR ALGEBRA CALCULATION')
    log('=' * 60)
    log(f'Image size = {w} x {h}')

    mp_pose = mp.solutions.pose

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=True
    ) as pose:
        result = pose.process(img_rgb)

    if not result.pose_landmarks or result.segmentation_mask is None:
        print('ไม่พบ pose landmarks / person mask')
        return 0, None, {
            'status': 0,
            'message': 'pose_or_person_mask_not_found',
            'left_arm': 0,
            'right_arm': 0
        }

    lm = result.pose_landmarks.landmark
    L = mp_pose.PoseLandmark

    def pt(idx):
        return np.array(
            [lm[idx].x * w, lm[idx].y * h],
            dtype=np.float32
        )

    # Person Mask
    M_person = (result.segmentation_mask >= 0.30).astype(np.float32)

    k = 2 * FEATHER + 1

    M_person = cv2.dilate(
        M_person,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (k, k)
        )
    )

    M_person = np.clip(
        cv2.GaussianBlur(
            M_person,
            (0, 0),
            FEATHER / 3
        ),
        0,
        1
    )

    # Torso Protection
    torso = np.array(
        [
            pt(L.LEFT_SHOULDER),
            pt(L.RIGHT_SHOULDER),
            pt(L.RIGHT_HIP),
            pt(L.LEFT_HIP)
        ],
        dtype=np.int32
    )

    M_torso = np.zeros((h, w), np.float32)
    cv2.fillPoly(M_torso, [torso], 1.0)

    M_torso = cv2.dilate(
        M_torso,
        np.ones((7, 7), np.uint8)
    )

    M_torso = np.clip(
        cv2.GaussianBlur(
            M_torso,
            (7, 7),
            0
        ),
        0,
        1
    )

    arms = {
        'LEFT ARM': (
            L.LEFT_SHOULDER,
            L.LEFT_ELBOW,
            L.LEFT_WRIST
        ),
        'RIGHT ARM': (
            L.RIGHT_SHOULDER,
            L.RIGHT_ELBOW,
            L.RIGHT_WRIST
        ),
    }

    P = pixel_grid(h, w)
    fields = []

    arm_status = {
        'LEFT ARM': 0,
        'RIGHT ARM': 0
    }

    for name, (i_s, i_e, i_w) in arms.items():

        vis = [lm[i].visibility for i in (i_s, i_e, i_w)]

        upper_ok = (
            vis[0] >= VISIBILITY_THRESHOLD
            and vis[1] >= VISIBILITY_THRESHOLD
        )

        lower_ok = (
            vis[1] >= VISIBILITY_THRESHOLD
            and vis[2] >= VISIBILITY_THRESHOLD
        )

        log()
        log('=' * 60)
        log(name)
        log('=' * 60)
        log(f'visibility = {[round(v, 4) for v in vis]}')
        log(f'upper = {"detect" if upper_ok else "skip"}')
        log(f'lower = {"detect" if lower_ok else "skip"}')

        if not (upper_ok or lower_ok):
            arm_status[name] = 0
            continue

        S, E, Wr = pt(i_s), pt(i_e), pt(i_w)

        log(f'Shoulder S = {fmt_vec(S)}')
        log(f'Elbow    E = {fmt_vec(E)}')
        log(f'Wrist    W = {fmt_vec(Wr)}')

        if upper_ok:
            report_segment(
                name + ' Shoulder->Elbow',
                S,
                E
            )
            report_projection_example(
                name + ' upper',
                S,
                E,
                UPPER_RADIUS
            )

        if lower_ok:
            report_segment(
                name + ' Elbow->Wrist',
                E,
                Wr
            )
            report_projection_example(
                name + ' lower',
                E,
                Wr,
                LOWER_RADIUS
            )

        field = build_arm_field(
            h,
            w,
            S,
            E,
            Wr,
            upper_ok,
            lower_ok
        )

        G_before = field['G']

        G_final = (
            G_before
            * M_person
            * (1.0 - M_torso)
        )

        field['G'] = G_final.astype(np.float32)

        log()
        log('Hadamard Product')
        log('G_final = G ⊙ M_person ⊙ (1-M_torso)')
        log(f'max(G)       = {float(np.max(G_before)):.4f}')
        log(f'max(G_final) = {float(np.max(G_final)):.4f}')
        log(f'mean(G_final)= {float(np.mean(G_final)):.6f}')

        # มีพื้นที่ mask จริงหลังตัด person/torso หรือไม่
        has_mask = bool(np.any(G_final > 1e-4))

        if has_mask:
            arm_status[name] = 1
            fields.append(field)
        else:
            arm_status[name] = 0

    # ไม่มีแขนข้างใดสร้าง mask ได้
    if not fields:
        print('มองไม่เห็นแขนทั้งสองข้าง / สร้าง mask ไม่ได้')

        return 0, None, {
            'status': 0,
            'message': 'arm_mask_not_found',
            'left_arm': arm_status['LEFT ARM'],
            'right_arm': arm_status['RIGHT ARM']
        }

    level = int(np.clip(reshape_level, -50, 50))
    scale = 1.0 - level / 100.0

    report_scale(level, scale)

    out = reshape_arms(
        img,
        P,
        fields,
        scale
    )

    log()
    log('=' * 60)
    log('VECTOR ADDITION + INVERSE MAPPING')
    log('=' * 60)
    log('V_total(P) = Σ V_i(P)')
    log('P_source   = P + V_total(P)')
    log('จากนั้นประมาณค่าสีด้วย Bilinear Interpolation จาก 4 pixel รอบข้าง')

    cv2.imwrite(OUTPUT_PATH, out)

    with open(
        CALCULATION_REPORT_PATH,
        'w',
        encoding='utf-8'
    ) as f:
        f.write('\n'.join(CALC_LINES))

    print()
    print('PROCESS COMPLETE')
    print(f'RESHAPE_LEVEL = {level:+d}')
    print(f'scale = {scale:.4f}')
    print('result:', OUTPUT_PATH)
    print('calculation:', CALCULATION_REPORT_PATH)

    return 1, out, {
        'status': 1,
        'message': 'success',
        'left_arm': arm_status['LEFT ARM'],
        'right_arm': arm_status['RIGHT ARM'],
        'reshape_level': level,
        'scale': float(scale)
    }


def main():
    """
    สำหรับรันไฟล์นี้โดยตรง
    """
    status, out, info = process_arm(
        IMAGE_PATH,
        RESHAPE_LEVEL
    )

    print()
    print('MASK STATUS =', status)
    print('INFO =', info)

    return status


if __name__ == '__main__':
    main()
