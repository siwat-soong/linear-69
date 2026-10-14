import cv2
import numpy as np
from PIL import Image, ImageOps
from insightface.app import FaceAnalysis

# ==========================================
# 1. โหลดโมเดล InsightFace / RetinaFace
# ==========================================
# name='buffalo_sc' เป็นโมเดลตรวจจับใบหน้าขนาดเล็ก แม่นยำ และทำงานรวดเร็ว
# ctx_id=-1 สำหรับประมวลผลบน CPU (หากมี GPU NVIDIA ให้เปลี่ยนเป็น ctx_id=0)
app = FaceAnalysis(name='buffalo_sc', allowed_modules=['detection'])
app.prepare(ctx_id=-1, det_size=(640, 640))


# ==========================================
# 2. ฟังก์ชันจัดการภาพและคณิตศาสตร์การสร้าง Mask
# ==========================================
def read_image_with_exif_fix(image_path):
    """
    อ่านภาพและจัดการ EXIF Orientation ให้ภาพอยู่แนวตั้งที่ถูกต้องก่อนประมวลผล
    """
    try:
        pil_img = Image.open(image_path)
        pil_img = ImageOps.exif_transpose(pil_img)
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img
    except Exception:
        return cv2.imread(image_path)


def create_2d_gaussian_kernel(kernel_size=3, sigma=1.0):
    """
    สร้าง Gaussian Kernel สำหรับเกลี่ยขอบ Mask
    """
    k_half = kernel_size // 2
    x = np.arange(-k_half, k_half + 1)
    X_k, Y_k = np.meshgrid(x, x)
    K_2d = np.exp(-(X_k**2 + Y_k**2) / (2.0 * sigma**2))
    return K_2d / np.sum(K_2d)


def apply_2d_convolution(M_binary, K_2d):
    """
    ทำ Gaussian Convolution กับ Binary Mask
    """
    return cv2.filter2D(
        M_binary,
        cv2.CV_32F,
        K_2d,
        borderType=cv2.BORDER_REFLECT
    )


def create_nose_polygon(
    p_nose_bridge,
    p_nostril_right,
    p_nostril_left,
    p_eye_right,
    p_eye_left,
    d_IPD
):
    """
    สร้างจุดยอด 4 จุดของ Nose Protection Mask โดยคำนวณทิศทางหมุนตามใบหน้าจริง (Rotation-Aware)
    """
    p_nose_bridge = np.array(p_nose_bridge, dtype=np.float64)
    p_nostril_right = np.array(p_nostril_right, dtype=np.float64)
    p_nostril_left = np.array(p_nostril_left, dtype=np.float64)
    p_eye_right = np.array(p_eye_right, dtype=np.float64)
    p_eye_left = np.array(p_eye_left, dtype=np.float64)

    # คำนวณเวกเตอร์ทิศทางตา (u_vec) และแนวตั้งฉากลงล่าง (v_vec)
    eye_vec = p_eye_left - p_eye_right
    norm_eye = np.linalg.norm(eye_vec)

    if norm_eye == 0:
        u_vec = np.array([1.0, 0.0], dtype=np.float64)
    else:
        u_vec = eye_vec / norm_eye

    v_vec = np.array([-u_vec[1], u_vec[0]], dtype=np.float64)

    # คำนวณพิกัดจุด N1 - N4
    N1 = p_nose_bridge - 0.15 * d_IPD * u_vec
    N2 = p_nose_bridge + 0.15 * d_IPD * u_vec
    N3 = p_nostril_right + 0.10 * d_IPD * u_vec + 0.05 * d_IPD * v_vec
    N4 = p_nostril_left - 0.10 * d_IPD * u_vec + 0.05 * d_IPD * v_vec

    return np.array([N1, N2, N3, N4], dtype=np.float64)


def create_nose_mask(
    image_shape,
    p_eye_right,
    p_eye_left,
    p_nose_bridge,
    p_nostril_right,
    p_nostril_left,
    kernel_size=3,
    sigma=1.0
):
    """
    สร้าง Nose Protection Mask แบบ Soft Mask
    """
    H, W = image_shape[:2]

    p_eye_right = np.asarray(p_eye_right, dtype=np.float64)
    p_eye_left = np.asarray(p_eye_left, dtype=np.float64)

    d_IPD = np.linalg.norm(p_eye_right - p_eye_left)

    if d_IPD <= 0:
        raise ValueError("ไม่สามารถคำนวณ d_IPD ได้")

    polygon_points = create_nose_polygon(
        p_nose_bridge,
        p_nostril_right,
        p_nostril_left,
        p_eye_right,
        p_eye_left,
        d_IPD
    )

    polygon = np.round(polygon_points).astype(np.int32)
    M_binary = np.zeros((H, W), dtype=np.float32)

    cv2.fillPoly(M_binary, [polygon], color=1.0)

    K_2d = create_2d_gaussian_kernel(kernel_size=kernel_size, sigma=sigma)
    M_nose = apply_2d_convolution(M_binary, K_2d)
    M_nose = np.clip(M_nose, 0.0, 1.0)

    return M_nose, M_binary, polygon


# ==========================================
# 3. ฟังก์ชันตรวจจับตำแหน่งจมูกด้วย InsightFace
# ==========================================
def mask_nose(image_path, kernel_size=3, sigma=1.0):
    """
    ตรวจจับ Landmark ด้วย RetinaFace (InsightFace) แล้วสร้าง Nose Protection Mask
    """
    img = read_image_with_exif_fix(image_path)

    if img is None:
        raise FileNotFoundError(f"ไม่พบไฟล์ภาพ: {image_path}")

    H, W = img.shape[:2]

    # ตรวจจับใบหน้า
    faces = app.get(img)

    # หากตรวจไม่พบใบหน้า คืน Zero Mask
    if len(faces) == 0:
        M_nose = np.zeros((H, W), dtype=np.float32)
        M_display = np.zeros((H, W), dtype=np.uint8)
        return M_nose, M_display, img

    # ดึง Keypoints 5 จุดของใบหน้าที่มีค่า Detection Score สูงสุด
    # kps = [ตาขวา, ตาซ้าย, ปลายจมูก, มุมปากขวา, มุมปากซ้าย]
    kps = faces[0].kps

    p_eye_right = kps[0].astype(np.float64)
    p_eye_left = kps[1].astype(np.float64)
    p_nose_tip = kps[2].astype(np.float64)

    d_IPD = np.linalg.norm(p_eye_left - p_eye_right)

    # ประเมินจุดสันจมูก (กึ่งกลางระหว่างตา ลงมาหาปลายจมูก)
    p_eye_mid = (p_eye_right + p_eye_left) / 2.0
    p_nose_bridge = p_eye_mid + 0.35 * (p_nose_tip - p_eye_mid)

    # คำนวณเวกเตอร์แนวนอนตามแนวตา
    eye_vec = p_eye_left - p_eye_right
    norm_eye = np.linalg.norm(eye_vec)
    u_vec = eye_vec / norm_eye if norm_eye > 0 else np.array([1.0, 0.0])

    # ประเมินจุดปีกจมูกขวา-ซ้าย
    p_nostril_right = p_nose_tip - 0.15 * d_IPD * u_vec
    p_nostril_left = p_nose_tip + 0.15 * d_IPD * u_vec

    # สร้าง Mask
    M_nose, M_binary, polygon = create_nose_mask(
        img.shape,
        p_eye_right,
        p_eye_left,
        p_nose_bridge,
        p_nostril_right,
        p_nostril_left,
        kernel_size=kernel_size,
        sigma=sigma
    )

    M_display = (np.clip(M_nose, 0.0, 1.0) * 255).astype(np.uint8)

    return M_nose, M_display, img


# ==========================================
# 4. ฟังก์ชัน Warp ปรับขนาดจมูก
# ==========================================
def nose_warp(img, M_nose, amount, ky=0.35, rx_k=0.9, ry_k=0.5, cy_pos=0.65):
    """
    ทำการ Warp ภาพตามบริเวณ Nose Mask
    """
    H, W = img.shape[:2]
    ys, xs = np.where(M_nose > 0.1)
    if len(xs) == 0:
        return img.copy()

    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    nose_w, nose_h = x1 - x0 + 1, y1 - y0 + 1

    cx = (x0 + x1) / 2.0
    cy = y0 + cy_pos * nose_h
    Rx, Ry = rx_k * nose_w, ry_k * nose_h

    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    dx, dy = gx - cx, gy - cy

    u = (dx / Rx) ** 2 + (dy / Ry) ** 2
    w = np.where(u < 1.0, (1.0 - u) ** 2, 0.0).astype(np.float32)

    a = float(amount)
    
    # คำนวณ Map และแปลง dtype เป็น np.float32
    map_x = (gx - dx * a * w).astype(np.float32)
    map_y = (gy - dy * a * ky * w).astype(np.float32)

    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101)

def resize_nose(image_path, amount=0.25, kernel_size=3, sigma=1.0):
    M_nose, M_display, img = mask_nose(
        image_path,
        kernel_size=kernel_size,
        sigma=sigma
    )

    if M_nose is None or not np.any(M_nose > 0.1):
        return img.copy(), M_nose

    result = nose_warp(
        img=img,
        M_nose=M_nose,
        amount=amount,
        ky=0.35,
        rx_k=0.9,
        ry_k=0.5,
        cy_pos=0.65
    )

    return result, M_nose


# ==========================================
# 5. การทดสอบการใช้งาน
# ==========================================
if __name__ == "__main__":
    image_path = "dataset/img_0071_jpeg.jpg"
    amount = -0.8  # ค่าบวก = ขยาย, ค่าลบ = หด

    result, nose_mask = resize_nose(
        image_path,
        amount=amount,
        kernel_size=3,
        sigma=1.0
    )

    original = read_image_with_exif_fix(image_path)

    cv2.imshow("Original Image", original)
    cv2.imshow("Nose Mask", nose_mask)
    cv2.imshow("Resized Nose", result)

    cv2.imwrite("nose_adjusted.jpg", result)
    print("Saved: nose_adjusted.jpg")

    cv2.waitKey(0)
    cv2.destroyAllWindows()