import cv2
import numpy as np
from PIL import Image, ImageOps
from insightface.app import FaceAnalysis

# ==========================================
# 1. โหลดโมเดล InsightFace
# ==========================================
app = FaceAnalysis(name='buffalo_sc', allowed_modules=['detection'])
app.prepare(ctx_id=-1, det_size=(640, 640))


# ==========================================
# 2. ฟังก์ชันจัดการภาพและคณิตศาสตร์ Mask
# ==========================================
def read_image_with_exif_fix(image_path):
    try:
        pil_img = Image.open(image_path)
        pil_img = ImageOps.exif_transpose(pil_img)
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img
    except Exception:
        return cv2.imread(image_path)


def create_2d_gaussian_kernel(kernel_size=9, sigma=3.0):
    k_half = kernel_size // 2
    x = np.arange(-k_half, k_half + 1)
    X_k, Y_k = np.meshgrid(x, x)
    K_2d = np.exp(-(X_k**2 + Y_k**2) / (2.0 * sigma**2))
    return K_2d / np.sum(K_2d)


def apply_2d_convolution(M_binary, K_2d):
    return cv2.filter2D(
        M_binary,
        cv2.CV_32F,
        K_2d,
        borderType=cv2.BORDER_REFLECT
    )


def create_eyebrow_polygons(c_brow_right, c_brow_left, u_vec, v_vec, d_IPD):
    """
    สร้างกรอบ Polygon ครอบคิ้วขวาและซ้าย
    """
    poly_right = np.array([
        c_brow_right + 0.35 * d_IPD * u_vec - 0.12 * d_IPD * v_vec,
        c_brow_right - 0.35 * d_IPD * u_vec - 0.12 * d_IPD * v_vec,
        c_brow_right - 0.35 * d_IPD * u_vec + 0.12 * d_IPD * v_vec,
        c_brow_right + 0.35 * d_IPD * u_vec + 0.12 * d_IPD * v_vec
    ], dtype=np.float64)

    poly_left = np.array([
        c_brow_left - 0.35 * d_IPD * u_vec - 0.12 * d_IPD * v_vec,
        c_brow_left + 0.35 * d_IPD * u_vec - 0.12 * d_IPD * v_vec,
        c_brow_left + 0.35 * d_IPD * u_vec + 0.12 * d_IPD * v_vec,
        c_brow_left - 0.35 * d_IPD * u_vec + 0.12 * d_IPD * v_vec
    ], dtype=np.float64)

    return poly_right, poly_left


# ==========================================
# 3. ฟังก์ชันครอปใบหน้า (Face ROI) แล้วคำนวณคิ้ว
# ==========================================
def mask_eyebrows_from_crop(image_path, kernel_size=9, sigma=3.0):
    img = read_image_with_exif_fix(image_path)

    if img is None:
        raise FileNotFoundError(f"ไม่พบไฟล์ภาพ: {image_path}")

    H, W = img.shape[:2]

    # 1. ตรวจจับใบหน้าเพื่อหา Bounding Box
    faces = app.get(img)

    if len(faces) == 0:
        M_eyebrows = np.zeros((H, W), dtype=np.float32)
        M_display = np.zeros((H, W), dtype=np.uint8)
        return M_eyebrows, M_display, img, None, None, 0

    # ดึง Bounding Box [x1, y1, x2, y2]
    bbox = faces[0].bbox.astype(int)
    x1, y1 = max(0, bbox[0]), max(0, bbox[1])
    x2, y2 = min(W, bbox[2]), min(H, bbox[3])

    # 2. ครอปภาพเฉพาะส่วนใบหน้า (Face ROI)
    face_crop = img[y1:y2, x1:x2]

    # 3. แปลง Keypoints จากภาพใหญ่เป็นพิกัดภายใน Face Crop (Local Coordinates)
    kps = faces[0].kps
    p_eye_right_local = kps[0].astype(np.float64) - np.array([x1, y1])
    p_eye_left_local = kps[1].astype(np.float64) - np.array([x1, y1])

    d_IPD = np.linalg.norm(p_eye_left_local - p_eye_right_local)
    if d_IPD <= 0:
        d_IPD = 1.0

    # คำนวณเวกเตอร์ความเอียงของใบหน้าภายใน Crop
    eye_vec = p_eye_left_local - p_eye_right_local
    norm_eye = np.linalg.norm(eye_vec)
    u_vec = eye_vec / norm_eye if norm_eye > 0 else np.array([1.0, 0.0])
    v_vec = np.array([-u_vec[1], u_vec[0]], dtype=np.float64)

    # 4. หาศูนย์กลางคิ้วขวา-ซ้าย เหนือดวงตาในพิกัด Local
    c_brow_right_local = p_eye_right_local - 0.22 * d_IPD * v_vec
    c_brow_left_local = p_eye_left_local - 0.22 * d_IPD * v_vec

    # 5. แปลงพิกัดกลับสู่ภาพใหญ่ (Global Coordinates)
    c_brow_right = c_brow_right_local + np.array([x1, y1])
    c_brow_left = c_brow_left_local + np.array([x1, y1])

    # 6. สร้าง Mask บนภาพใหญ่
    poly_right, poly_left = create_eyebrow_polygons(c_brow_right, c_brow_left, u_vec, v_vec, d_IPD)

    M_binary = np.zeros((H, W), dtype=np.float32)
    cv2.fillPoly(M_binary, [np.round(poly_right).astype(np.int32)], color=1.0)
    cv2.fillPoly(M_binary, [np.round(poly_left).astype(np.int32)], color=1.0)

    K_2d = create_2d_gaussian_kernel(kernel_size=kernel_size, sigma=sigma)
    M_eyebrows = apply_2d_convolution(M_binary, K_2d)
    M_eyebrows = np.clip(M_eyebrows, 0.0, 1.0)

    M_display = (np.clip(M_eyebrows, 0.0, 1.0) * 255).astype(np.uint8)

    return M_eyebrows, M_display, img, c_brow_right, c_brow_left, d_IPD


# ==========================================
# 4. ฟังก์ชัน Warp ปรับขนาดคิ้ว
# ==========================================
def eyebrows_warp(img, c_brow_right, c_brow_left, d_IPD, amount, ky=0.7, rx_k=0.40, ry_k=0.25):
    H, W = img.shape[:2]

    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))

    Rx = rx_k * d_IPD
    Ry = ry_k * d_IPD

    # ปรับเรโชใหม่: ให้ amount = 1.0 ได้แรงดึง boost = 0.75 (เท่ากับ 0.15 เดิม)
    boost = 0.75 * float(amount)


    # 1. แรงดึงคิ้วขวา
    dx_r = gx - c_brow_right[0]
    dy_r = gy - c_brow_right[1]
    u_r = (dx_r / Rx) ** 2 + (dy_r / Ry) ** 2
    w_r = np.where(u_r < 1.0, (1.0 - u_r) ** 2, 0.0).astype(np.float32)

    # 2. แรงดึงคิ้วซ้าย
    dx_l = gx - c_brow_left[0]
    dy_l = gy - c_brow_left[1]
    u_l = (dx_l / Rx) ** 2 + (dy_l / Ry) ** 2
    w_l = np.where(u_l < 1.0, (1.0 - u_l) ** 2, 0.0).astype(np.float32)

    # คำนวณ Map
    map_x = (gx - dx_r * boost * w_r - dx_l * boost * w_l).astype(np.float32)
    map_y = (gy - dy_r * boost * ky * w_r - dy_l * boost * ky * w_l).astype(np.float32)

    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101)


def resize_eyebrows(image_path, amount=0.50, kernel_size=9, sigma=3.0):
    M_eyebrows, M_display, img, c_brow_right, c_brow_left, d_IPD = mask_eyebrows_from_crop(
        image_path,
        kernel_size=kernel_size,
        sigma=sigma
    )

    if c_brow_right is None or d_IPD <= 0:
        return img.copy(), M_eyebrows

    result = eyebrows_warp(
        img=img,
        c_brow_right=c_brow_right,
        c_brow_left=c_brow_left,
        d_IPD=d_IPD,
        amount=amount,
        ky=0.70,
        rx_k=0.40,
        ry_k=0.25
    )

    return result, M_eyebrows


# ==========================================
# 5. การทดสอบการใช้งาน
# ==========================================
if __name__ == "__main__":
    image_path = "dataset/img_0097_jpeg.jpg"

    # amount > 0 = ขยายคิ้วหนา/ใหญ่ขึ้น, amount < 0 = ลดขนาดคิ้วบางลง
    amount = -1.0

    result, eyebrow_mask = resize_eyebrows(
        image_path,
        amount=amount,
        kernel_size=9,
        sigma=3.0
    )

    original = read_image_with_exif_fix(image_path)

    cv2.imshow("Original Image", original)
    cv2.imshow("Eyebrow Mask", eyebrow_mask)
    cv2.imshow("Resized Eyebrows", result)

    cv2.imwrite("eyebrows_adjusted.jpg", result)
    print("Saved: eyebrows_adjusted.jpg")

    cv2.waitKey(0)
    cv2.destroyAllWindows()