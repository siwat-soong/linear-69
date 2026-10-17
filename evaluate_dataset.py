import os
import sys
import glob
import cv2
import pandas as pd

# =====================================================================
# 1. จัดการ Path ของโปรเจกต์ (แทรก Path ทุกระดับชั้นป้องกัน ImportError)
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.join(BASE_DIR, "main")
UTILS_DIR = os.path.join(BASE_DIR, "main", "utils")
MASK_DIR = os.path.join(BASE_DIR, "main", "utils", "mask")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

for path in [BASE_DIR, MAIN_DIR, UTILS_DIR, MASK_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

# =====================================================================
# 2. Import ฟังก์ชัน Reshape Matrix
# =====================================================================
try:
    from reshape_matrix import resize_max_640_conv
except ImportError as e:
    print(f"⚠️ ไม่สามารถ Import reshape_matrix ได้: {e}")
    def resize_max_640_conv(img):
        return img

# =====================================================================
# 3. Import ฟังก์ชัน Validate จากทุกโมดูล
# =====================================================================
try:
    from mask_eyebrow import validate_brow
    from mask_eye import validate_eye
    from mask_nose import validate_nose
    from mask_mouth import validate_mouth
    from mask_cheek import validate_cheek
    from chest import validate_chest
    from waist import validate_waist
    from hips import validate_hips

    try:
        from mask_arm import validate_arm
    except ImportError:
        from arm_for_ui import validate_arm

    try:
        from mask_leg import validate_leg
    except ImportError:
        from leg_for_ui import validate_leg

except ImportError as e:
    print("=" * 65)
    print(f"❌ เกิดข้อผิดพลาดในการ Import โมดูล Validate: {e}")
    print(f"📍 ตรวจสอบโฟลเดอร์: {MASK_DIR}")
    print("=" * 65)
    sys.exit(1)

VALIDATORS = {
    "Eyebrow": validate_brow,
    "Eye": validate_eye,
    "Nose": validate_nose,
    "Mouth": validate_mouth,
    "Cheek": validate_cheek,
    "Chest": validate_chest,
    "Waist": validate_waist,
    "Hips": validate_hips,
    "Arm": validate_arm,
    "Leg": validate_leg
}

os.makedirs(REPORTS_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(REPORTS_DIR, "dataset_evaluation_report.csv")
OUTPUT_EXCEL = os.path.join(REPORTS_DIR, "dataset_evaluation_report.xlsx")
SUPPORTED_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")


def run_evaluation():
    image_paths = []
    for ext in SUPPORTED_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(DATASET_DIR, ext)))
        image_paths.extend(glob.glob(os.path.join(DATASET_DIR, ext.upper())))

    image_paths = sorted(list(set(image_paths)))
    total_images = len(image_paths)

    if total_images == 0:
        print(f"⚠️ ไม่พบรูปภาพในโฟลเดอร์ '{DATASET_DIR}'")
        return

    print("=" * 65)
    print(f"🚀 เริ่มต้นประมวลผลการทดสอบ Dataset จำนวนทั้งหมด {total_images} รูป")
    print(f"📂 ตำแหน่งภาพ Dataset : {DATASET_DIR}")
    print("=" * 65)

    report_data = []
    organ_pass_counts = {organ: 0 for organ in VALIDATORS.keys()}
    
    reshape_needed_count = 0
    valid_for_warp_count = 0

    for idx, img_path in enumerate(image_paths, 1):
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)

        if img is None:
            row = {
                "No.": idx,
                "Filename": filename,
                "Original_Size": "N/A",
                "Is_Reshape_Needed": "N/A",
                "Processed_Size": "N/A",
                "Read_Status": "Corrupted/Unreadable",
                **{organ: 0 for organ in VALIDATORS.keys()},
                "Valid_Organs_Count": 0,
                "Is_Valid_For_Warp": "NO"
            }
            report_data.append(row)
            print(f"[{idx}/{total_images}] ❌ อ่านรูปภาพไม่ได้: {filename}")
            continue

        orig_h, orig_w = img.shape[:2]
        max_dim = max(orig_h, orig_w)

        # 1. เช็กความจำเป็นในการ Reshape (ถ้าน้อยกว่า หรือ มากกว่า 640px ถือว่าจำเป็นทั้งหมด)
        is_reshape_needed = "YES" if max_dim != 640 else "NO"
        if is_reshape_needed == "YES":
            reshape_needed_count += 1

        # 2. นำภาพไปผ่าน reshape_matrix
        processed_img = resize_max_640_conv(img)
        proc_h, proc_w = processed_img.shape[:2]

        row = {
            "No.": idx,
            "Filename": filename,
            "Original_Size": f"{orig_w}x{orig_h}",
            "Is_Reshape_Needed": is_reshape_needed,
            "Processed_Size": f"{proc_w}x{proc_h}",
            "Read_Status": "OK"
        }

        valid_count = 0

        # 3. Validate แต่ละอวัยวะ
        for organ_name, validate_fn in VALIDATORS.items():
            try:
                status = validate_fn(processed_img)
                status = 1 if status == 1 else 0
            except Exception as e:
                print(f"⚠️ Error validating {organ_name} on {filename}: {e}")
                status = 0

            row[organ_name] = status
            if status == 1:
                valid_count += 1
                organ_pass_counts[organ_name] += 1

        row["Valid_Organs_Count"] = valid_count
        is_valid_for_warp = "YES" if valid_count > 0 else "NO"
        row["Is_Valid_For_Warp"] = is_valid_for_warp

        if valid_count > 0:
            valid_for_warp_count += 1

        report_data.append(row)
        print(f"[{idx}/{total_images}] 📸 {filename} ({orig_w}x{orig_h}) | Reshape Needed: {is_reshape_needed} | Valid Organs: {valid_count}/{len(VALIDATORS)}")

    # บันทึกเป็น CSV และ Excel
    df = pd.DataFrame(report_data)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    excel_saved = False
    try:
        df.to_excel(OUTPUT_EXCEL, index=False, engine="openpyxl")
        excel_saved = True
    except Exception:
        pass

    # =====================================================================
    # 4. แสดงผลสรุปสถิติท้ายการทำงาน
    # =====================================================================
    print("\n" + "=" * 65)
    print("📊 ผลการสรุปสถิติการทดสอบ (DATASET EVALUATION SUMMARY REPORT)")
    print("=" * 65)
    print(f"จำนวนรูปภาพทั้งหมดที่ตรวจสอบ : {total_images} รูป")
    print("-" * 65)
    print("1. สถิติความจำเป็นในการ Reshape ภาพ (ขนาดไม่เท่ากับ 640px):")
    print(f"   - จำเป็นต้อง Reshape (< หรือ > 640px) : {reshape_needed_count}/{total_images} รูป ({(reshape_needed_count/total_images)*100:.2f}%)")
    print(f"   - ขนาดพอดีมาตรฐาน (เท่ากับ 640px)     : {total_images - reshape_needed_count}/{total_images} รูป ({((total_images - reshape_needed_count)/total_images)*100:.2f}%)")
    print("-" * 65)
    print("2. สถิติความพร้อมในการใช้งาน (Validation Pass):")
    print(f"   - ผ่านอย่างน้อย 1 อวัยวะ (พร้อม Warp) : {valid_for_warp_count}/{total_images} รูป ({(valid_for_warp_count/total_images)*100:.2f}%)")
    print(f"   - ไม่ผ่านเลยทุกอวัยวะ                 : {total_images - valid_for_warp_count}/{total_images} รูป")
    print("-" * 65)
    print("3. รายละเอียดจำนวนรูปที่ผ่าน Validation แยกตามอวัยวะ:")
    print("-" * 65)

    for organ, count in organ_pass_counts.items():
        percentage = (count / total_images) * 100 if total_images > 0 else 0
        print(f" - {organ:<10} : ผ่าน {count:>4}/{total_images} รูป ({percentage:>6.2f}%)")

    print("=" * 65)
    print(f"📁 บันทึกไฟล์รายงานสำเร็จเรียบร้อย:")
    print(f"   - CSV File   : {OUTPUT_CSV}")
    if excel_saved:
        print(f"   - Excel File : {OUTPUT_EXCEL}")
    print("=" * 65)


if __name__ == "__main__":
    run_evaluation()