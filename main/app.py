from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware  # 1. Import CORSMiddleware
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import numpy as np
import hashlib
import cv2
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # อนุญาตทุก Origin (รวมถึง 'null' จาก file://)
    allow_credentials=True,
    allow_methods=["*"],        # อนุญาตทุก HTTP Method (POST, GET, etc.)
    allow_headers=["*"],        # อนุญาตทุก Headers
)

@app.post("/reshape_matrix")
async def reshape_matrix(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        matrix = np.array(pil_image)

        from utils.reshape_matrix import resize_max_640_conv
        cleaned_matrix = resize_max_640_conv(matrix)
        
        result_image = Image.fromarray(cleaned_matrix.astype(np.uint8))
        
        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        
        return Response(content=img_bytes, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")


from utils.mask.mask_eye import validate_eye, create_eye_mask, eye_adjust
from utils.mask.mask_mouth import validate_mouth, create_mouth_mask, mouth_adjust
from utils.mask.mask_cheek import validate_cheek, create_cheek_mask, cheek_adjust
from utils.mask.mask_eyebrow import validate_brow, create_brow_mask, brow_adjust
from utils.mask.mask_nose import validate_nose, create_nose_mask, nose_adjust
from utils.mask.chest import validate_chest, create_chest_mask, chest_adjust
from utils.mask.waist import validate_waist, create_waist_mask, waist_adjust
from utils.mask.hips import validate_hips, create_hips_mask, hips_adjust
from utils.mask.arm_for_ui import validate_arm, create_arm_mask, arm_adjust
from utils.mask.leg_for_ui import validate_leg, create_leg_mask, leg_adjust

WARP_CACHE = {}

def get_image_hash(image_bytes: bytes) -> str:
    return hashlib.md5(image_bytes).hexdigest()

@app.get("/")
async def serve_ui():
    return FileResponse("ui_linear.html")

# 💡 2. เปิดให้เข้าถึงไฟล์ Static Assets เพิ่มเติม (ถ้ามี)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


# Dictionary สำหรับรวบรวมฟังก์ชัน Validate
VALIDATORS = {
    "brow": validate_brow,
    "eye": validate_eye,
    "nose": validate_nose,
    "mouth": validate_mouth,  # เพิ่ม validate ปาก
    "cheek": validate_cheek,
    "chest": validate_chest,
    "waist": validate_waist,
    "hips": validate_hips,
    "arm": validate_arm,
    "leg": validate_leg
}

# =====================================================================
# Validate Endpoint
# =====================================================================
@app.post("/validate")
async def validate(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        matrix = np.array(pil_image)

        valid_parts = []
        for key, validator_fn in VALIDATORS.items():
            if validator_fn(matrix):
                valid_parts.append(key)

        return valid_parts

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Validation failed: {str(e)}")


@app.post("/eye/mask")
async def eye_mask_endpoint(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        img_hash = get_image_hash(contents)
        cache_key = f"{img_hash}_eye"

        nparr = np.frombuffer(contents, np.uint8)
        matrix = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        M_eye_display, V_warp = create_eye_mask(matrix)

        if V_warp is None:
            raise ValueError("ไม่พบดวงตาในภาพเพื่อทำ Mask")

        WARP_CACHE[cache_key] = V_warp

        success, encoded_mask = cv2.imencode('.png', M_eye_display)
        if not success:
            raise ValueError("Failed to encode mask image")

        return Response(content=encoded_mask.tobytes(), media_type="image/png")

    except Exception as e:
        print(f"Error in /eye/mask: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/eye")
async def eye_adjust_endpoint(image: UploadFile = File(...), scale: float = Form(0.0)):
    try:
        contents = await image.read()
        img_hash = get_image_hash(contents)
        cache_key = f"{img_hash}_eye"

        nparr = np.frombuffer(contents, np.uint8)
        matrix = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if matrix is None:
            raise ValueError("ไม่สามารถอ่านข้อมูลรูปภาพได้")

        if cache_key in WARP_CACHE:
            V_warp = WARP_CACHE[cache_key]
        else:
            _, V_warp = create_eye_mask(matrix)
            if V_warp is not None:
                WARP_CACHE[cache_key] = V_warp
            else:
                raise ValueError("ยังไม่ได้ทำ Mask หรือหาดวงตาไม่พบ")

        alpha = float(scale) / 100.0
        result_matrix = eye_adjust(matrix, V_warp, alpha=alpha)

        success, encoded_img = cv2.imencode('.png', result_matrix)
        if not success:
            raise ValueError("Failed to encode image")

        return Response(content=encoded_img.tobytes(), media_type="image/png")

    except Exception as e:
        print(f"Error in /eye: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/mouth/mask")
async def mouth_mask_endpoint(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        img_hash = get_image_hash(contents)
        cache_key = f"{img_hash}_mouth"

        nparr = np.frombuffer(contents, np.uint8)
        matrix = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        M_mouth_display, V_warp = create_mouth_mask(matrix)

        if V_warp is None:
            raise ValueError("ไม่พบมุมปากในภาพเพื่อทำ Mask")

        WARP_CACHE[cache_key] = V_warp

        success, encoded_mask = cv2.imencode('.png', M_mouth_display)
        if not success:
            raise ValueError("Failed to encode mask image")

        return Response(content=encoded_mask.tobytes(), media_type="image/png")

    except Exception as e:
        print(f"Error in /mouth/mask: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/mouth")
async def mouth_adjust_endpoint(image: UploadFile = File(...), scale: float = Form(0.0)):
    try:
        contents = await image.read()
        img_hash = get_image_hash(contents)
        cache_key = f"{img_hash}_mouth"

        nparr = np.frombuffer(contents, np.uint8)
        matrix = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if matrix is None:
            raise ValueError("ไม่สามารถอ่านข้อมูลรูปภาพได้")

        if cache_key in WARP_CACHE:
            V_warp = WARP_CACHE[cache_key]
        else:
            _, V_warp = create_mouth_mask(matrix)
            if V_warp is not None:
                WARP_CACHE[cache_key] = V_warp
            else:
                raise ValueError("ยังไม่ได้ทำ Mask หรือหามุมปากไม่พบ")

        alpha = float(scale) / 100.0
        result_matrix = mouth_adjust(matrix, V_warp, alpha=alpha)

        success, encoded_img = cv2.imencode('.png', result_matrix)
        if not success:
            raise ValueError("Failed to encode image")

        return Response(content=encoded_img.tobytes(), media_type="image/png")

    except Exception as e:
        print(f"Error in /mouth: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/cheek/mask")
async def cheek_mask_endpoint(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        img_hash = get_image_hash(contents)
        cache_key = f"{img_hash}_cheek"

        nparr = np.frombuffer(contents, np.uint8)
        matrix = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        M_cheek_display, V_warp = create_cheek_mask(matrix)

        if V_warp is None:
            raise ValueError("ไม่พบใบหน้า/แก้มในภาพเพื่อทำ Mask")

        WARP_CACHE[cache_key] = V_warp

        success, encoded_mask = cv2.imencode('.png', M_cheek_display)
        if not success:
            raise ValueError("Failed to encode mask image")

        return Response(content=encoded_mask.tobytes(), media_type="image/png")

    except Exception as e:
        print(f"Error in /cheek/mask: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/cheek")
async def cheek_adjust_endpoint(image: UploadFile = File(...), scale: float = Form(0.0)):
    try:
        contents = await image.read()
        img_hash = get_image_hash(contents)
        cache_key = f"{img_hash}_cheek"

        nparr = np.frombuffer(contents, np.uint8)
        matrix = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if cache_key in WARP_CACHE:
            V_warp = WARP_CACHE[cache_key]
        else:
            _, V_warp = create_cheek_mask(matrix)
            if V_warp is not None:
                WARP_CACHE[cache_key] = V_warp
            else:
                raise ValueError("ยังไม่ได้ทำ Mask หรือหาแก้มไม่พบ")

        alpha = float(scale) / -100.0
        result_matrix = cheek_adjust(matrix, V_warp, alpha=alpha)

        success, encoded_img = cv2.imencode('.png', result_matrix)
        if not success:
            raise ValueError("Failed to encode image")

        return Response(content=encoded_img.tobytes(), media_type="image/png")

    except Exception as e:
        print(f"Error in /cheek: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/brow/mask")
async def brow_mask_endpoint(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        img_hash = get_image_hash(contents)
        cache_key = f"{img_hash}_brow"

        nparr = np.frombuffer(contents, np.uint8)
        matrix = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        M_brow_display, V_warp = create_brow_mask(matrix)

        if V_warp is None:
            raise ValueError("ไม่พบใบหน้า/คิ้วในภาพเพื่อทำ Mask")

        WARP_CACHE[cache_key] = V_warp

        success, encoded_mask = cv2.imencode('.png', M_brow_display)
        if not success:
            raise ValueError("Failed to encode mask image")

        return Response(content=encoded_mask.tobytes(), media_type="image/png")

    except Exception as e:
        print(f"Error in /brow/mask: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/brow")
async def brow_adjust_endpoint(image: UploadFile = File(...), scale: float = Form(0.0)):
    try:
        contents = await image.read()
        img_hash = get_image_hash(contents)
        cache_key = f"{img_hash}_brow"

        nparr = np.frombuffer(contents, np.uint8)
        matrix = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if matrix is None:
            raise ValueError("ไม่สามารถอ่านข้อมูลรูปภาพได้")

        if cache_key in WARP_CACHE:
            V_warp = WARP_CACHE[cache_key]
        else:
            _, V_warp = create_brow_mask(matrix)
            if V_warp is not None:
                WARP_CACHE[cache_key] = V_warp
            else:
                raise ValueError("ยังไม่ได้ทำ Mask หรือหาคิ้วไม่พบ")

        alpha = float(scale) / 100.0
        result_matrix = brow_adjust(matrix, V_warp, alpha=alpha)

        success, encoded_img = cv2.imencode('.png', result_matrix)
        if not success:
            raise ValueError("Failed to encode image")

        return Response(content=encoded_img.tobytes(), media_type="image/png")

    except Exception as e:
        print(f"Error in /brow: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/nose/mask")
async def nose_mask_endpoint(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        img_hash = get_image_hash(contents)
        cache_key = f"{img_hash}_nose"

        nparr = np.frombuffer(contents, np.uint8)
        matrix = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        M_nose_display, V_warp = create_nose_mask(matrix)

        if V_warp is None:
            raise ValueError("ไม่พบใบหน้า/จมูกในภาพเพื่อทำ Mask")

        WARP_CACHE[cache_key] = V_warp

        success, encoded_mask = cv2.imencode('.png', M_nose_display)
        if not success:
            raise ValueError("Failed to encode mask image")

        return Response(content=encoded_mask.tobytes(), media_type="image/png")

    except Exception as e:
        print(f"Error in /nose/mask: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/nose")
async def nose_adjust_endpoint(image: UploadFile = File(...), scale: float = Form(0.0)):
    try:
        contents = await image.read()
        img_hash = get_image_hash(contents)
        cache_key = f"{img_hash}_nose"

        nparr = np.frombuffer(contents, np.uint8)
        matrix = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if matrix is None:
            raise ValueError("ไม่สามารถอ่านข้อมูลรูปภาพได้")

        if cache_key in WARP_CACHE:
            V_warp = WARP_CACHE[cache_key]
        else:
            _, V_warp = create_nose_mask(matrix)
            if V_warp is not None:
                WARP_CACHE[cache_key] = V_warp
            else:
                raise ValueError("ยังไม่ได้ทำ Mask หรือหาจมูกไม่พบ")

        alpha = float(scale) / 100.0
        result_matrix = nose_adjust(matrix, V_warp, alpha=alpha)

        success, encoded_img = cv2.imencode('.png', result_matrix)
        if not success:
            raise ValueError("Failed to encode image")

        return Response(content=encoded_img.tobytes(), media_type="image/png")

    except Exception as e:
        print(f"Error in /nose: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Helper Generic Endpoint Builder สำหรับส่วนลำตัว
async def handle_torso_mask(image: UploadFile, part_name: str, mask_fn):
    contents = await image.read()
    img_hash = get_image_hash(contents)
    cache_key = f"{img_hash}_{part_name}"

    nparr = np.frombuffer(contents, np.uint8)
    matrix = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    M_display, body_cache_data = mask_fn(matrix)
    if body_cache_data is None:
        raise HTTPException(status_code=400, detail=f"ไม่พบโครงสร้างลำตัวเพื่อทำ Mask ({part_name})")

    WARP_CACHE[cache_key] = body_cache_data
    success, encoded_mask = cv2.imencode('.png', M_display)
    return Response(content=encoded_mask.tobytes(), media_type="image/png")

async def handle_torso_adjust(image: UploadFile, scale: float, part_name: str, mask_fn, adjust_fn):
    contents = await image.read()
    img_hash = get_image_hash(contents)
    cache_key = f"{img_hash}_{part_name}"

    nparr = np.frombuffer(contents, np.uint8)
    matrix = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if cache_key in WARP_CACHE:
        body_cache_data = WARP_CACHE[cache_key]
    else:
        _, body_cache_data = mask_fn(matrix)
        if body_cache_data is not None:
            WARP_CACHE[cache_key] = body_cache_data
        else:
            raise HTTPException(status_code=400, detail=f"ไม่พบข้อมูล Mask ({part_name})")

    result_matrix = adjust_fn(matrix, body_cache_data, scale_val=-scale)
    success, encoded_img = cv2.imencode('.png', result_matrix)
    return Response(content=encoded_img.tobytes(), media_type="image/png")

# Endpoints Chest
@app.post("/chest/mask")
async def chest_mask_endpoint(image: UploadFile = File(...)):
    return await handle_torso_mask(image, "chest", create_chest_mask)

@app.post("/chest")
async def chest_adjust_endpoint(image: UploadFile = File(...), scale: float = Form(0.0)):
    return await handle_torso_adjust(image, scale, "chest", create_chest_mask, chest_adjust)

# Endpoints Waist
@app.post("/waist/mask")
async def waist_mask_endpoint(image: UploadFile = File(...)):
    return await handle_torso_mask(image, "waist", create_waist_mask)

@app.post("/waist")
async def waist_adjust_endpoint(image: UploadFile = File(...), scale: float = Form(0.0)):
    return await handle_torso_adjust(image, scale, "waist", create_waist_mask, waist_adjust)

# Endpoints Hips
@app.post("/hips/mask")
async def hips_mask_endpoint(image: UploadFile = File(...)):
    return await handle_torso_mask(image, "hips", create_hips_mask)

@app.post("/hips")
async def hips_adjust_endpoint(image: UploadFile = File(...), scale: float = Form(0.0)):
    return await handle_torso_adjust(image, scale, "hips", create_hips_mask, hips_adjust)

@app.post("/arm/mask")
async def arm_mask_endpoint(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        img_hash = get_image_hash(contents)
        cache_key = f"{img_hash}_arm"

        nparr = np.frombuffer(contents, np.uint8)
        matrix = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        M_arm_display, arm_cache_data = create_arm_mask(matrix)

        if arm_cache_data is None:
            raise ValueError("ไม่พบแขนในภาพเพื่อทำ Mask")

        WARP_CACHE[cache_key] = arm_cache_data

        success, encoded_mask = cv2.imencode('.png', M_arm_display)
        if not success:
            raise ValueError("Failed to encode mask image")

        return Response(content=encoded_mask.tobytes(), media_type="image/png")

    except Exception as e:
        print(f"Error in /arm/mask: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/arm")
async def arm_adjust_endpoint(image: UploadFile = File(...), scale: float = Form(0.0)):
    try:
        contents = await image.read()
        img_hash = get_image_hash(contents)
        cache_key = f"{img_hash}_arm"

        nparr = np.frombuffer(contents, np.uint8)
        matrix = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if matrix is None:
            raise ValueError("ไม่สามารถอ่านข้อมูลรูปภาพได้")

        if cache_key in WARP_CACHE:
            arm_cache_data = WARP_CACHE[cache_key]
        else:
            _, arm_cache_data = create_arm_mask(matrix)
            if arm_cache_data is not None:
                WARP_CACHE[cache_key] = arm_cache_data
            else:
                raise ValueError("ยังไม่ได้ทำ Mask หรือหาแขนไม่พบ")

        result_matrix = arm_adjust(matrix, arm_cache_data, scale_val=-scale)

        success, encoded_img = cv2.imencode('.png', result_matrix)
        if not success:
            raise ValueError("Failed to encode image")

        return Response(content=encoded_img.tobytes(), media_type="image/png")

    except Exception as e:
        print(f"Error in /arm: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/leg/mask")
async def leg_mask_endpoint(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        img_hash = get_image_hash(contents)
        cache_key = f"{img_hash}_leg"

        nparr = np.frombuffer(contents, np.uint8)
        matrix = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        M_leg_display, leg_cache_data = create_leg_mask(matrix)

        if leg_cache_data is None:
            raise ValueError("ไม่พบขาในภาพเพื่อทำ Mask")

        WARP_CACHE[cache_key] = leg_cache_data

        success, encoded_mask = cv2.imencode('.png', M_leg_display)
        if not success:
            raise ValueError("Failed to encode mask image")

        return Response(content=encoded_mask.tobytes(), media_type="image/png")

    except Exception as e:
        print(f"Error in /leg/mask: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/leg")
async def leg_adjust_endpoint(image: UploadFile = File(...), scale: float = Form(0.0)):
    try:
        contents = await image.read()
        img_hash = get_image_hash(contents)
        cache_key = f"{img_hash}_leg"

        nparr = np.frombuffer(contents, np.uint8)
        matrix = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if matrix is None:
            raise ValueError("ไม่สามารถอ่านข้อมูลรูปภาพได้")

        if cache_key in WARP_CACHE:
            leg_cache_data = WARP_CACHE[cache_key]
        else:
            _, leg_cache_data = create_leg_mask(matrix)
            if leg_cache_data is not None:
                WARP_CACHE[cache_key] = leg_cache_data
            else:
                raise ValueError("ยังไม่ได้ทำ Mask หรือหาขาไม่พบ")

        result_matrix = leg_adjust(matrix, leg_cache_data, scale_val=-scale)

        success, encoded_img = cv2.imencode('.png', result_matrix)
        if not success:
            raise ValueError("Failed to encode image")

        return Response(content=encoded_img.tobytes(), media_type="image/png")

    except Exception as e:
        print(f"Error in /leg: {e}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)