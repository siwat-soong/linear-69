FROM python:3.10-slim

# ติดตั้ง Linux System Libraries
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

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir tensorflow-cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Pre-cache InsightFace Model
RUN python -c "import insightface; app = insightface.app.FaceAnalysis(name='buffalo_l'); app.prepare(ctx_id=-1, det_size=(640, 640))"

COPY . .

EXPOSE 8000

ENV PYTHONPATH=/app/main

CMD ["uvicorn", "main.app:app", "--host", "0.0.0.0", "--port", "8000"]