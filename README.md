# Body Reshaping Project (Linear 69)

ระบบประมวลผลและปรับแต่งโครงสร้างอวัยวะร่างกายด้วยเทคโนโลยี Computer Vision & Deep Learning (FastAPI + MediaPipe + PyTorch + InsightFace)

---

## 🛠️ ข้อกำหนดก่อนติดตั้ง (Prerequisites)
* ติดตั้ง [Docker Desktop](https://www.docker.com/products/docker-desktop/) บนเครื่องคอมพิวเตอร์ และเปิดโปรแกรมให้พร้อมใช้งาน

---

## 🚀 ขั้นตอนการติดตั้งและเปิดใช้งานครั้งแรก

### 1. คลองโปรเจกต์ (Clone Repository)
```bash
git clone https://github.com/siwat-soong/linear-69.git
cd linear-69
```

### 2. บิลด์และสั่งรันระบบ (Build & Run - ครั้งแรกเท่านั้น)
เปิด Terminal ในโฟลเดอร์โปรเจกต์ แล้วรันคำสั่ง:

```bash
docker compose up --build
```
*(การรันครั้งแรกอาจใช้เวลา 3-5 นาที ในการติดตั้ง Dependencies และดาวน์โหลด Weights)*

---

## 🔄 การเปิดใช้งานในครั้งถัดไป (ไม่ต้องบิลด์ใหม่)

หากเคยบิลด์ระบบผ่านแล้ว ในการใช้งานครั้งถัดไป **ไม่จำเป็นต้องสั่ง `--build` ซ้ำ** สามารถทำได้ดังนี้:

### 1. ปิดแค่หน้าเว็บ (แต่ Terminal ยังเปิดรันอยู่)
เปิด Web Browser แล้วเข้า URL เดิมได้ทันที:
👉 **[http://localhost:8000](http://localhost:8000)**

### 2. สั่งเปิดระบบใหม่ (หลังปิด Terminal หรือสั่ง Stop ไปแล้ว)
เปิด Terminal ในโฟลเดอร์โปรเจกต์ แล้วสั่ง:
```bash
docker compose up
```
*(ระบบจะพร้อมใช้งานภายใน 5–10 วินาที)*

> 💡 **ต้องการรันแบบซ่อนไว้เบื้องหลัง (Background Mode):**
> ```bash
> docker compose up -d
> ```

### 3. เปิดผ่านโปรแกรม Docker Desktop (GUI)
1. เปิดโปรแกรม **Docker Desktop**
2. ไปที่เมนู **Containers**
3. กดปุ่ม ▶️ **(Start)** ที่ Container ชื่อ `body_reshape_container`

---

## 🛑 การหยุดการทำงานของระบบ (Stop System)

* **หากรันหน้า Terminal:** กด `Ctrl + C`
* **หากรันแบบ Background หรือต้องการปิดระบบสมบูรณ์:**
  ```bash
  docker compose down
  ```

---

## 🌐 เข้าใช้งานระบบ

เมื่อโปรเซสใน Terminal ขึ้นสถานะ `SERVER IS READY!` ให้เปิดเว็บเบราว์เซอร์แล้วเข้าไปที่:

👉 **[http://localhost:8000](http://localhost:8000)**

> ⚠️ **ข้อควรระวัง (Important Note):**
> แม้ใน Terminal ของ Docker/Uvicorn จะแสดงข้อความ `http://0.0.0.0:8000` **ห้ามคัดลอก `0.0.0.0` ไปวางใน Web Browser** เด็ดขาด เพราะระบบปฏิบัติการ (โดยเฉพาะ Windows) จะบล็อกการเชื่อมต่อ ให้ใช้ **`http://localhost:8000`** หรือ **`http://127.0.0.1:8000`** เท่านั้น

## 🛠️ รายชื่อระบบประมวลผลที่ใช้ภายใน (AI Engines)
* **MediaPipe:** Pose Landmark Detection & Face Mesh
* **InsightFace & ONNX Runtime:** Face Analysis & Landmark Triangulation
* **PyTorch (FAN):** 3D Face Alignment
* **MTCNN:** Multi-task Cascaded Convolutional Networks
* **OpenCV & SciPy:** Image Warping & Mask Morphology