import numpy as np

def build_1d_separable_kernel(src_len: int, dst_len: int) -> np.ndarray:
    s = dst_len / src_len
    # ปรับรัศมี Kernel เพื่อป้องกัน Anti-aliasing เมื่อย่อภาพ
    scale = min(1.0, s)
    radius = 1.0 / scale
    
    # คำนวณพิกัดจุดกึ่งกลางของภาพใหม่ที่แมปกลับไปยังภาพเดิม
    dst_coords = np.arange(dst_len)
    src_coords = (dst_coords + 0.5) / s - 0.5
    
    # คำนวณระยะห่างระหว่างพิกัดภาพเดิมและจุดสุ่มใหม่
    src_indices = np.arange(src_len)[:, None]  # Shape: (src_len, 1)
    dist = np.abs(src_indices - src_coords[None, :])  # Shape: (src_len, dst_len)
    
    # คำนวณค่า Convolution Weight ด้วย Triangle (Bilinear) Kernel
    weights = np.maximum(0.0, 1.0 - dist / radius)
    
    # Normalize น้ำหนักให้ผลรวมของแต่ละพิกเซลเป้าหมายเท่ากับ 1.0
    weight_sums = weights.sum(axis=0, keepdims=True)
    weight_sums[weight_sums == 0] = 1.0
    weights /= weight_sums
    
    return weights  # Shape: (src_len, dst_len)


def resize_max_640_conv(image: np.ndarray, target_max: int = 640) -> np.ndarray:
    H, W = image.shape[:2]
    max_dim = max(H, W)
    
    # หากขนาดตรงตามเป้าหมายอยู่แล้ว ไม่ต้องคำนวณซ้ำ
    if max_dim == target_max:
        return image.copy()
        
    # 1. คำนวณขนาดภาพใหม่โดยคงสัดส่วนเดิม (Aspect Ratio)
    s = target_max / max_dim
    if H >= W:
        H_new = target_max
        W_new = int(round(W * s))
    else:
        W_new = target_max
        H_new = int(round(H * s))

    # 2. สร้าง Convolution Weight Matrices ในแต่ละแกน
    # Wy: 1D Kernel ในแกน Y (แนวตั้ง) ขนาด (H_new, H)
    Wy = build_1d_separable_kernel(H, H_new).T 
    # Wx: 1D Kernel ในแกน X (แนวนอน) ขนาด (W, W_new)
    Wx = build_1d_separable_kernel(W, W_new)

    # 3. ประมวลผล Separable 2D Convolution: I_new = Wy @ Image @ Wx
    if image.ndim == 2:  # ภาพ Grayscale (2D Matrix)
        image_resized = Wy @ image.astype(np.float32) @ Wx
    else:  # ภาพ Color (3D Matrix - H x W x Channels)
        image_resized = np.zeros((H_new, W_new, image.shape[2]), dtype=np.float32)
        for c in range(image.shape[2]):
            image_resized[:, :, c] = Wy @ image[:, :, c].astype(np.float32) @ Wx

    # Clipping ค่าความเข้มสีให้อยู่ในช่วง [0, 255]
    return np.clip(image_resized, 0, 255).astype(image.dtype)