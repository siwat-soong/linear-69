# บีบ/ขยาย "อก" ด้วย Vector-Matrix (scale_val: -50..+50, +50=เล็กลง -50=ใหญ่ขึ้น)
#
# แยกเป็น 2 ฟังก์ชันสำหรับใช้ในแอพ (ตอนอัพโหลดภาพ ทำครั้งเดียว vs ตอนผู้ใช้ลากสไลเดอร์
# ปรับค่า ทำซ้ำ ๆ แบบ real-time) เพราะ pose detection + สร้าง mask ใช้เวลาหลักวินาที
# แต่ขั้นตอนบิดภาพใช้แค่หลักสิบมิลลิวินาที ถ้าไม่แยกจะทำให้สไลเดอร์หน่วงทุกครั้งที่ลาก
from body_common import prepare_body_data, reshape_single_part

T_CENTER_CHEST = 0.22
SIGMA_CHEST = 0.09


def chest_prepare(image_path):
    """ขั้นตอนตอนอัพโหลดไฟล์ (ทำครั้งเดียว) : หา pose landmark + สร้าง mask ลำตัว"""
    return prepare_body_data(image_path)


def chest_adjust(data, scale_val=0):
    """ขั้นตอนตอนผู้ใช้ปรับค่า (เรียกซ้ำได้ real-time) : รับ data จาก chest_prepare()
    มาบีบ/ขยายเฉพาะ 'อก' เท่านั้น เอว/สะโพกไม่ถูกแตะต้อง"""
    return reshape_single_part(
        data['img'], data['P'], data['Q'], data['t_raw'], data['G_body'],
        T_CENTER_CHEST, SIGMA_CHEST, scale_val,
    )


if __name__ == "__main__":
    import cv2

    IMAGE_PATH = './dataset/img_0221_jpeg.jpg'
    data = chest_prepare(IMAGE_PATH)          # ทำครั้งเดียวตอนอัพโหลด
    out = chest_adjust(data, scale_val=25)    # เรียกซ้ำได้เร็ว ตอนปรับสไลเดอร์
    cv2.imwrite('./experiment/phase2/chest_result.png', out)
    print("บันทึกผลลัพธ์ที่ ./experiment/phase2/chest_result.png")
