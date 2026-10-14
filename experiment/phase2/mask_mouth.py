import cv2
import numpy as np
from mtcnn import MTCNN

def create_2d_gaussian_kernel(kernel_size=21, sigma=5.0):
    k_half = kernel_size // 2
    x = np.arange(-k_half, k_half + 1)
    X_k, Y_k = np.meshgrid(x, x)
    K_2d = np.exp(-(X_k**2 + Y_k**2) / (2.0 * (sigma**2)))
    return K_2d / np.sum(K_2d)

def apply_2d_convolution(M_binary, K_2d):
    return cv2.filter2D(M_binary, -1, K_2d, borderType=cv2.BORDER_REFLECT)

def bilinear_interpolate_vectorized(img, P_src):
    H, W, C = img.shape
    X_src, Y_src = P_src[..., 0], P_src[..., 1]
    
    x0 = np.clip(np.floor(X_src).astype(np.int32), 0, W - 1)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y0 = np.clip(np.floor(Y_src).astype(np.int32), 0, H - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    
    wx = (X_src - x0)[..., np.newaxis]
    wy = (Y_src - y0)[..., np.newaxis]
    
    top = (1.0 - wx) * img[y0, x0] + wx * img[y0, x1]
    bottom = (1.0 - wx) * img[y1, x0] + wx * img[y1, x1]
    
    return np.clip((1.0 - wy) * top + wy * bottom, 0, 255).astype(np.uint8)

def process_mouth_warping(img, p_MR, p_ML, scale_val=0, alpha=0.3, kernel_size=25, sigma_conv=6.0):
    H, W, _ = img.shape
    p_MR, p_ML = np.array(p_MR, dtype=np.float64), np.array(p_ML, dtype=np.float64)

    # 1. Coordinate Grid & Displacement Vector เทียบกับจุดศูนย์กลางปาก
    X_grid, Y_grid = np.meshgrid(np.arange(W), np.arange(H))
    P_grid = np.stack([X_grid, Y_grid], axis=-1).astype(np.float64)
    
    p_M = (p_MR + p_ML) / 2.0  # จุดศูนย์กลางปาก
    v_M = P_grid - p_M         # เวกเตอร์ระยะห่างจากจุดศูนย์กลางปาก

    # 2. Rotation Matrix (Change of Basis ตามมุมเอียงของปาก)
    diff = p_ML - p_MR
    theta = np.arctan2(diff[1], diff[0])
    R_theta = np.array([
        [ np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ], dtype=np.float64)

    v_prime_M = v_M @ R_theta.T

    # 3. Quadratic Form Distance Matrix (สัดส่วนวงรีของปาก)
    d_mouth = np.linalg.norm(diff, ord=2)
    a_m, b_m = 0.60 * d_mouth, 0.30 * d_mouth  # ปรับสัดส่วนกว้าง x สูง ให้ครอบคลุมริมฝีปาก
    k = 1.0 + (scale_val / 100.0)
    
    A_scaled = (1.0 / (k**2)) * np.diag([1.0 / (a_m**2), 1.0 / (b_m**2)])

    E_M = np.sum((v_prime_M @ A_scaled) * v_prime_M, axis=-1)

    # 4. 2D Gaussian Kernel Matrix Convolution (สร้าง Soft Mask ปาก)
    M_hard_M = np.where(E_M <= 1.0, 1.0, 0.0)
    K_2d = create_2d_gaussian_kernel(kernel_size=kernel_size, sigma=sigma_conv)
    M_M = apply_2d_convolution(M_hard_M, K_2d)[..., np.newaxis]

    # 5. Vector Displacement Field & Backward Mapping
    V_warp = M_M * v_M
    P_src = P_grid - (alpha * V_warp)

    # 6. Bilinear Interpolation
    warped_img = bilinear_interpolate_vectorized(img, P_src)
    M_display = (np.clip(M_M[..., 0], 0.0, 1.0) * 255).astype(np.uint8)

    return warped_img, M_display

def mouth_adjust(image_path, scale_val=0, alpha=0.0):
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: ไม่พบไฟล์ภาพที่ '{image_path}'")
        exit()
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    results = detector.detect_faces(img_rgb)
    
    if not results:
        print("ไม่พบใบหน้าในรูปภาพ!")
        exit()
    
    # ดึงพิกัดมุมปากจาก MTCNN
    keypoints = results[0]['keypoints']
    p_MR = keypoints['mouth_right']  # พิกัดมุมปากขวา [x, y]
    p_ML = keypoints['mouth_left']   # พิกัดมุมปากซ้าย [x, y]
    
    img_warped, _ = process_mouth_warping(
        img, p_MR, p_ML, scale_val=scale_val, alpha=alpha, kernel_size=25, sigma_conv=5.0
    )

    return img_warped