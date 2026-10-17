import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # ปิด PyTorch Compiler ป้องกัน Error เรื่อง cl.exe[cite: 1]

import cv2
import face_alignment
import numpy as np
from scipy.signal import convolve2d

# =========================================================
# Step 1: Face Midline & Direction Vector (แก้ไขเรื่อง Vectorization)[cite: 1]
# =========================================================
def calculate_face_midline(p_nose, p_chin):
    v_midline = p_chin - p_nose
    norm_sq = np.dot(v_midline, v_midline)
    return v_midline, norm_sq

def project_point_to_line(p, p_nose, v_midline, norm_sq):
    a = p - p_nose
    # ใช้ np.sum + keepdims=True เพื่อรองรับทั้งจุดเดียว และ Matrix P_grid (H, W, 2)[cite: 1]
    c = np.sum(a * v_midline, axis=-1, keepdims=True) / norm_sq
    p_proj = p_nose + c * v_midline
    return p_proj

def calculate_displacement_direction(p, p_proj):
    d = p_proj - p
    # ใส่ axis=-1, keepdims=True เพื่อหา Norm บนแกนเวกเตอร์ (x, y)[cite: 1]
    distance = np.linalg.norm(d, axis=-1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        u = np.where(distance == 0, 0.0, d / distance)
    return d, np.squeeze(distance, axis=-1), u

def face_midline_vector(P_nose, P_chin, p):
    v_midline, norm_sq = calculate_face_midline(P_nose, P_chin)
    p_proj = project_point_to_line(p, P_nose, v_midline, norm_sq)
    d, distance, u = calculate_displacement_direction(p, p_proj)

    return d, distance, u


# =========================================================
# Step 2: Soft Mask / Weight Map Generation[cite: 1]
# =========================================================
def create_2d_gaussian_kernel(kernel_size=15, sigma=5.0):
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(- (xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

def generate_soft_mask(image_shape, cheek_landmarks, kernel_size=15, sigma=5.0):
    binary_mask = np.zeros(image_shape, dtype=np.float32)

    for pt in cheek_landmarks:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= y < image_shape[0] and 0 <= x < image_shape[1]: 
            binary_mask[y, x] = 1.0

    kernel = create_2d_gaussian_kernel(kernel_size, sigma)
    soft_mask = convolve2d(binary_mask, kernel, mode='same', boundary='symm')
    
    max_val = np.max(soft_mask)
    if max_val > 0:
        soft_mask = soft_mask / max_val

    return soft_mask


# =========================================================
# Step 3: Displacement Field Matrix Generation (แก้ไขชื่อตัวแปร soft_mask_W)[cite: 1]
# =========================================================
def generate_displacement_field(image_shape, p_nose, p_chin, soft_mask_W, k=10.0):
    h, w = image_shape  # ตัวแปร w ตรงนี้จะไม่ซ้ำกับ soft_mask_W แล้ว[cite: 1]

    x_coord = np.arange(w, dtype=np.float32)
    y_coord = np.arange(h, dtype=np.float32)
    X, Y = np.meshgrid(x_coord, y_coord)
    P_grid = np.stack([X, Y], axis=-1)

    _, _, U_grid = face_midline_vector(p_nose, p_chin, P_grid)

    W_expanded = np.expand_dims(soft_mask_W, axis=-1)
    displacement_field = k * W_expanded * U_grid

    return displacement_field


# =========================================================
# Step 4: Inverse Mapping & Bilinear Interpolation Warping[cite: 1]
# =========================================================
def bilinear_interpolate(image, x_indices, y_indices):
    h, w = image.shape[:2]

    x_clip = np.clip(x_indices, 0, w - 1 - 1e-4)
    y_clip = np.clip(y_indices, 0, h - 1 - 1e-4)

    x1 = np.floor(x_clip).astype(np.int32)
    y1 = np.floor(y_clip).astype(np.int32)
    x2 = np.minimum(x1 + 1, w - 1)
    y2 = np.minimum(y1 + 1, h - 1)

    dx = x_clip - x1
    dy = y_clip - y1

    if image.ndim == 3: 
        dx = np.expand_dims(dx, axis=-1)
        dy = np.expand_dims(dy, axis=-1)

    I_11 = image[y1, x1]
    I_21 = image[y1, x2]
    I_12 = image[y2, x1]
    I_22 = image[y2, x2]

    I_out = (1 - dx) * (1 - dy) * I_11 + \
            dx * (1 - dy) * I_21 + \
            (1 - dx) * dy * I_12 + \
            dx * dy * I_22
            
    return I_out


# =========================================================
# Helper & Main Execution[cite: 1]
# =========================================================
def extract_landmarks_from_image(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"ไม่พบไฟล์รูปภาพที่ตำแหน่ง: {image_path}")

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, 
        flip_input=False,
        device='cpu'
    )

    preds = fa.get_landmarks(image)
    if preds is None or len(preds) == 0:
        raise ValueError("ไม่พบใบหน้าในรูปภาพนี้")

    landmarks = preds[0]

    get_pt = lambda idx: np.array([landmarks[idx][0], landmarks[idx][1]], dtype=np.float32)

    p_nose = get_pt(30)
    p_chin = get_pt(8)

    cheek_indices = [2, 3, 4, 12, 13, 14]
    cheek_landmarks = [get_pt(idx) for idx in cheek_indices]

    return image, p_nose, p_chin, cheek_landmarks


if __name__ == "__main__":
    input_path = ''

    print('[O] Start')
    img, p_nose, p_chin, cheek_landmarks = extract_landmarks_from_image(input_path)
    print('[O] Extracted Image')
    h, w, _ = img.shape

    kernel_size = int(max(w, h) * 0.08) | 1
    sigma = kernel_size / 3.0
    W = generate_soft_mask((h, w), cheek_landmarks, kernel_size=kernel_size, sigma=sigma)
    print('[O] Weight Generated')

    k_strength = 10.0  # ปรับเพิ่ม/ลด แรงบีบได้ตรงนี้[cite: 1]
    D = generate_displacement_field((h, w), p_nose, p_chin, W, k=k_strength)
    print('[O] Field Generated')

    x_coords = np.arange(w, dtype=np.float32)
    y_coords = np.arange(h, dtype=np.float32)
    X, Y = np.meshgrid(x_coords, y_coords)
            
    X_src = X - D[:, :, 0]
    Y_src = Y - D[:, :, 1]
            
    warped_img = bilinear_interpolate(img.astype(np.float32), X_src, Y_src)
    warped_img = np.clip(warped_img, 0, 255).astype(np.uint8)
    print('[O] Done')

    cv2.imshow('warped', warped_img)
    cv2.imshow('original', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()