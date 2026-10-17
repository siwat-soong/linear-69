from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware  # 1. Import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
import io
import numpy as np
import hashlib
import cv2

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

WARP_CACHE = {}

def get_image_hash(image_bytes: bytes) -> str:
    return hashlib.md5(image_bytes).hexdigest()


# Dictionary สำหรับรวบรวมฟังก์ชัน Validate
VALIDATORS = {
    "brow": validate_brow,
    "eye": validate_eye,
    "mouth": validate_mouth,  # เพิ่ม validate ปาก
    "cheek": validate_cheek
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


# =====================================================================
# Endpoints สำหรับดวงตา (Eye)
# =====================================================================
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


# =====================================================================
# Endpoints สำหรับปาก (Mouth)
# =====================================================================
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

        alpha = float(scale) / 100.0
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)