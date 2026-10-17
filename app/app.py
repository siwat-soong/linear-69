from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # 1. Import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
import io
import numpy as np

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


from utils.mask.mask_eye import validate_eye
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)