FROM python:3.10-slim

# ติดตั้ง Linux System Libraries ที่จำเป็นสำหรับ OpenCV / MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. ติดตั้ง PyTorch CPU และ TensorFlow CPU
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir tensorflow-cpu

# 2. ติดตั้ง Requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Pre-cache InsightFace Models ป้องกันโหลดช้าตอนรันครั้งแรก
RUN python -c "import insightface; app = insightface.app.FaceAnalysis(name='buffalo_l'); app.prepare(ctx_id=-1, det_size=(640, 640))"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]