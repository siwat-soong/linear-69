# Body Reshaping Project (Linear 69)

ระบบประมวลผลและปรับแต่งโครงสร้างอวัยวะร่างกายด้วยเทคโนโลยี Computer Vision & Deep Learning (FastAPI + MediaPipe + PyTorch + InsightFace)

---

## 🚀 ขั้นตอนการติดตั้งและเปิดใช้งานด้วย Docker (แนะนำ)

### ข้อกำหนดก่อนติดตั้ง (Prerequisites)
* ติดตั้ง [Docker Desktop](https://www.docker.com/products/docker-desktop/) บนเครื่องคอมพิวเตอร์

---

### 1. คลองโปรเจกต์ (Clone Repository)
```bash
git clone [https://github.com/siwat-soong/linear-69.git](https://github.com/siwat-soong/linear-69.git)
cd linear-69
```

### 2. บิลด์และสั่งรันระบบ (Build & Run)
เปิด Terminal ในโฟลเดอร์โปรเจกต์ แล้วรันคำสั่ง:

```bash
docker compose up --build
```
*(การรันครั้งแรกอาจใช้เวลา 3-5 นาที ในการติดตั้ง Dependencies และดาวน์โหลด Weights)*

---

## 🌐 เข้าใช้งานระบบ

เมื่อโปรเซสใน Terminal ขึ้นสถานะ `Application startup complete` สามารถเปิดเว็บเบราว์เซอร์แล้วเข้าไปที่:

👉 **[http://localhost:8000](http://localhost:8000)**

---

## 🛠️ รายชื่อระบบประมวลผลที่ใช้ภายใน (AI Engines)
* **MediaPipe:** Pose Landmark Detection & Face Mesh
* **InsightFace & ONNX Runtime:** Face Analysis & Landmark Triangulation
* **PyTorch (FAN):** 3D Face Alignment
* **MTCNN:** Multi-task Cascaded Convolutional Networks
* **OpenCV & SciPy:** Image Warping & Mask Morphology