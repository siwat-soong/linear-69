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

def process_eye_warping(img, p_R, p_L, scale_val=0, alpha=0.3, kernel_size=25, sigma_conv=6.0):
    H, W, _ = img.shape
    p_R, p_L = np.array(p_R, dtype=np.float64), np.array(p_L, dtype=np.float64)

    # 1. Coordinate Grid & Displacement Fields
    X_grid, Y_grid = np.meshgrid(np.arange(W), np.arange(H))
    P_grid = np.stack([X_grid, Y_grid], axis=-1).astype(np.float64)
    v_R, v_L = P_grid - p_R, P_grid - p_L

    # 2. Rotation Matrix (Change of Basis)
    diff = p_L - p_R
    theta = np.arctan2(diff[1], diff[0])
    R_theta = np.array([
        [ np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ], dtype=np.float64)

    v_prime_R = v_R @ R_theta.T
    v_prime_L = v_L @ R_theta.T

    # 3. Quadratic Form Distance Matrix
    d_IPD = np.linalg.norm(diff, ord=2)
    a_e, b_e = 0.35 * d_IPD, 0.22 * d_IPD
    k = 1.0 + (scale_val / 100.0)
    
    A_scaled = (1.0 / (k**2)) * np.diag([1.0 / (a_e**2), 1.0 / (b_e**2)])

    E_R = np.sum((v_prime_R @ A_scaled) * v_prime_R, axis=-1)
    E_L = np.sum((v_prime_L @ A_scaled) * v_prime_L, axis=-1)

    # 4. 2D Gaussian Kernel Matrix Convolution
    M_hard_R = np.where(E_R <= 1.0, 1.0, 0.0)
    M_hard_L = np.where(E_L <= 1.0, 1.0, 0.0)

    K_2d = create_2d_gaussian_kernel(kernel_size=kernel_size, sigma=sigma_conv)

    M_R = apply_2d_convolution(M_hard_R, K_2d)[..., np.newaxis]
    M_L = apply_2d_convolution(M_hard_L, K_2d)[..., np.newaxis]

    # 5. Vector Displacement Field & Backward Mapping
    V_warp = (M_R * v_R) + (M_L * v_L)
    P_src = P_grid - (alpha * V_warp)

    # 6. Bilinear Interpolation
    warped_img = bilinear_interpolate_vectorized(img, P_src)

    # Mask Display Output
    M_eye_display = (np.clip(np.maximum(M_R[..., 0], M_L[..., 0]), 0.0, 1.0) * 255).astype(np.uint8)

    return warped_img, M_eye_display

def eye_adjust(image_path, scale_val=0, alpha=0.0):
        # 1. อ่านภาพต้นฉบับ
        img = cv2.imread(image_path)
        if img is None:
            print(f"ERROR: ไม่พบไฟล์ภาพที่ '{image_path}' กรุณาตรวจสอบชื่อไฟล์และพาธภาพ")
            exit()
    
        # 2. แปลง BGR เป็น RGB ก่อนส่งให้ MTCNN
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        print("กำลังตรวจจับใบหน้าและพิกัดดวงตาด้วย MTCNN...")
        detector = MTCNN()
        results = detector.detect_faces(img_rgb)
    
        if not results:
            print("ไม่พบใบหน้าในรูปภาพ!")
            exit()
    
        # 3. ดึงพิกัดดวงตาจาก MTCNN Landmark
        keypoints = results[0]['keypoints']
        p_R = keypoints['right_eye']  # พิกัดตาขวา [x, y]
        p_L = keypoints['left_eye']   # พิกัดตาซ้าย [x, y]
    
        img_warped, _ = process_eye_warping(
            img, p_R, p_L, scale_val=scale_val, alpha=alpha, kernel_size=25, sigma_conv=5.0
        )
        return img_warped