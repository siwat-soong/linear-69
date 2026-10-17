# ทดสอบ chest.py / waist.py / hips.py แบบ headless + เช็คว่าแต่ละไฟล์ปรับเฉพาะส่วนตัวเอง
# และเช็คว่าขั้นตอน prepare (ช้า) กับ adjust (เร็ว) แยกกันจริง เหมาะกับ slider real-time
import os
import time
import cv2
import numpy as np
from body_common import prepare_body_data, gaussian_band
from chest import chest_prepare, chest_adjust, T_CENTER_CHEST, SIGMA_CHEST
from waist import waist_prepare, waist_adjust, T_CENTER_WAIST, SIGMA_WAIST
from hips import hips_prepare, hips_adjust, T_CENTER_HIPS, SIGMA_HIPS

IMAGE_PATH = './dataset/img_0221_jpeg.jpg'
OUT_DIR = './experiment/phase2/out_test'
os.makedirs(OUT_DIR, exist_ok=True)

BAND = {
    'chest': (T_CENTER_CHEST, SIGMA_CHEST),
    'waist': (T_CENTER_WAIST, SIGMA_WAIST),
    'hips':  (T_CENTER_HIPS, SIGMA_HIPS),
}

# ---------- Phase 1 : prepare (ทำครั้งเดียวตอนอัพโหลด, ช้า เพราะมี pose detection) ----------
t0 = time.time()
d = chest_prepare(IMAGE_PATH)   # chest_prepare/waist_prepare/hips_prepare ให้ผลเหมือนกัน
t_prepare = time.time() - t0

img, t_raw, G_body = d['img'], d['t_raw'], d['G_body']
print("shoulder-L", d['L_sh'], "shoulder-R", d['R_sh'])
print("hip-L", d['L_hip'], "hip-R", d['R_hip'])
print(f"[timing] prepare (pose detect + mask) = {t_prepare*1000:.1f} ms  <- ทำครั้งเดียวตอนอัพโหลด")

cv2.imwrite(f"{OUT_DIR}/mask_person.png", (d['M_person'] * 255).astype(np.uint8))
cv2.imwrite(f"{OUT_DIR}/mask_torso_poly.png", (d['M_torso'] * 255).astype(np.uint8))

overlay = img.copy().astype(np.float32)
color = np.zeros_like(img, dtype=np.float32)
color[:, :, 2] = gaussian_band(t_raw, *BAND['chest']) * G_body * 255
color[:, :, 1] = gaussian_band(t_raw, *BAND['waist']) * G_body * 255
color[:, :, 0] = gaussian_band(t_raw, *BAND['hips']) * G_body * 255
overlay = cv2.addWeighted(overlay.astype(np.uint8), 1.0, color.astype(np.uint8), 0.6, 0)
cv2.line(overlay, tuple(d['Sc'].astype(int)), tuple(d['Hc'].astype(int)), (0, 255, 255), 2)
for p_ in (d['L_sh'], d['R_sh'], d['L_hip'], d['R_hip'], d['Sc'], d['Hc']):
    cv2.circle(overlay, tuple(p_.astype(int)), 5, (0, 255, 0), -1)
cv2.imwrite(f"{OUT_DIR}/overlay_bands.png", overlay)

# ---------- Phase 2 : adjust (เรียกซ้ำได้ real-time ตอนผู้ใช้ลากสไลเดอร์, ต้องเร็ว) ----------
t0 = time.time()
out_zero = chest_adjust(d, scale_val=0)
t_adjust = time.time() - t0
print(f"[timing] adjust (warp เดียว)          = {t_adjust*1000:.1f} ms  <- เรียกซ้ำได้ทุกครั้งที่ลากสไลเดอร์")
assert t_adjust < t_prepare / 5, "adjust ควรเร็วกว่า prepare มาก ไม่งั้น slider จะหน่วง"

diff = np.abs(img.astype(int) - out_zero.astype(int))
print(f"[sanity] scale_val=0 -> max diff={diff.max()} (ควรเป็น 0 หรือใกล้ 0)")
assert diff.max() <= 1, "scale_val=0 ควรให้ภาพเหมือนเดิม"

# ---------- ทดสอบความ "แยกอิสระ" ของแต่ละไฟล์/ฟังก์ชัน ----------
# บีบอกแรง ๆ (scale_val=+40) แล้ววัดว่าโซนเอว/สะโพก (t ไกลจาก t_chest) ไม่เปลี่ยนเลย
out_chest = chest_adjust(d, scale_val=40)
out_waist = waist_adjust(d, scale_val=40)
out_hips = hips_adjust(d, scale_val=40)

# พื้นที่ "ไกลจากแถบที่กำลังปรับ" (น้ำหนัก Gaussian ของแถบนั้น < 0.001 คือห่างจาก
# จุดศูนย์กลางแถบเกิน ~3.7 sigma) ต้องไม่เปลี่ยนเลย (ไม่นับ noise การปัดเศษ 8-bit)
far_from_chest = gaussian_band(t_raw, *BAND['chest']) < 0.001
far_from_waist = gaussian_band(t_raw, *BAND['waist']) < 0.001
far_from_hips = gaussian_band(t_raw, *BAND['hips']) < 0.001

d_chest = np.abs(img.astype(int) - out_chest.astype(int)).max(axis=-1)
d_waist = np.abs(img.astype(int) - out_waist.astype(int)).max(axis=-1)
d_hips = np.abs(img.astype(int) - out_hips.astype(int)).max(axis=-1)

# หมายเหตุ: Gaussian ไม่เคยเท่ากับ 0 เป๊ะ (ตั้งใจให้นุ่มนวล ไม่มีรอยตัดฉับพลัน
# ตามที่รายงานกำหนด) จึงยอมรับ noise การปัดเศษ 8-bit (<=1/255) ว่าคือ "ไม่ถูกแตะต้อง"
TOLERANCE = 1
print(f"[independence] chest.py: max diff นอกแถบอก = {d_chest[far_from_chest].max()} (ควร <= {TOLERANCE})")
print(f"[independence] waist.py: max diff นอกแถบเอว = {d_waist[far_from_waist].max()} (ควร <= {TOLERANCE})")
print(f"[independence] hips.py : max diff นอกแถบสะโพก = {d_hips[far_from_hips].max()} (ควร <= {TOLERANCE})")

assert d_chest[far_from_chest].max() <= TOLERANCE, "chest_adjust ไปแตะพื้นที่นอกแถบอกมากเกินไป!"
assert d_waist[far_from_waist].max() <= TOLERANCE, "waist_adjust ไปแตะพื้นที่นอกแถบเอวมากเกินไป!"
assert d_hips[far_from_hips].max() <= TOLERANCE, "hips_adjust ไปแตะพื้นที่นอกแถบสะโพกมากเกินไป!"

cv2.imwrite(f"{OUT_DIR}/result_chest_only_+40.png", out_chest)
cv2.imwrite(f"{OUT_DIR}/result_waist_only_+40.png", out_waist)
cv2.imwrite(f"{OUT_DIR}/result_hips_only_+40.png", out_hips)

# ---------- ทดสอบช่วงสเกล -50..+50 ทั้งสองขั้ว (เรียก adjust ซ้ำด้วย data เดิม ไม่ prepare ใหม่) ----------
for scale_val in (-50, -25, 0, 25, 50):
    out = waist_adjust(d, scale_val=scale_val)
    cv2.imwrite(f"{OUT_DIR}/waist_scale_{scale_val:+d}.png", out)
    print(f"saved waist_scale_{scale_val:+d}.png")

print("ALL TESTS PASSED, outputs in", OUT_DIR)
