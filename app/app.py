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
VALIDATORS = {
    "eye": validate_eye,
}

@app.post("/validate")
async def validate(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        matrix = np.array(pil_image)

        valid_parts = []

        for key, validator_fn in VALIDATORS.items():
            is_valid = validator_fn(matrix)
            if is_valid:
                valid_parts.append(key)

        return valid_parts

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Validation failed: {str(e)}")

WARP_CACHE = {}

def get_image_hash(image_bytes: bytes) -> str:
    return hashlib.md5(image_bytes).hexdigest()

@app.post("/eye/mask")
async def eye_mask_endpoint(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        img_hash = get_image_hash(contents)

        nparr = np.frombuffer(contents, np.uint8)
        matrix = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # คำนวณ Mask และ V_warp
        M_eye_display, V_warp = create_eye_mask(matrix)

        if V_warp is None:
            raise ValueError("ไม่พบดวงตาในภาพเพื่อทำ Mask")

        # บันทึก V_warp ลง Cache
        WARP_CACHE[img_hash] = V_warp

        # แปลงภาพ Mask (ขาว-ดำ) คืนกลับไปให้ Frontend แสดงผลพรีวิว (ถ้าต้องการ)
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

        # ดึง V_warp จาก Cache
        if img_hash not in WARP_CACHE:
            # หาก Cache หลุด ให้ทำการ Mask ใหม่ให้อัตโนมัติ (Fallback)
            nparr = np.frombuffer(contents, np.uint8)
            matrix = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            _, V_warp = create_eye_mask(matrix)
            if V_warp is not None:
                WARP_CACHE[img_hash] = V_warp
            else:
                raise ValueError("ยังไม่ได้ทำ Mask หรือหาดวงตาไม่พบ")
        else:
            V_warp = WARP_CACHE[img_hash]

            nparr = np.frombuffer(contents, np.uint8)
            matrix = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # คำนวณ alpha จาก scale
        alpha = float(scale) / 100.0

        # ปรับแต่งดวงตาด้วย eye_adjust (รวดเร็วมาก)
        result_matrix = eye_adjust(matrix, V_warp, alpha=alpha)

        success, encoded_img = cv2.imencode('.png', result_matrix)
        return Response(content=encoded_img.tobytes(), media_type="image/png")

    except Exception as e:
        print(f"Error in /eye: {e}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)