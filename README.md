# 📚 Classroom Attention Detection

Aplikasi berbasis Python untuk mendeteksi dan menganalisis fokus siswa di ruang kelas menggunakan **YOLOv8**, **MediaPipe**, dan **Streamlit**. Program ini mengidentifikasi wajah, melacak fokus, dan menghitung durasi fokus dan tidak fokus untuk setiap individu yang terdeteksi secara real-time melalui kamera.

---

<a href="https://ibb.co.com/kgwcF9bc"><img src="https://i.ibb.co.com/zhBfw79f/Whats-App-Image-2025-01-19-at-18-00-08-ec6d299b.jpg" alt="Whats-App-Image-2025-01-19-at-18-00-08-ec6d299b" border="0"></a>


## ✨ Fitur

- **Integrasi YOLOv8**: Menggunakan YOLOv8 untuk deteksi wajah yang efisien.
- **Analisis Fokus**: Menganalisis arah pandangan menggunakan MediaPipe FaceMesh untuk menentukan apakah seseorang fokus atau tidak.
- **Pelacakan Real-Time**: Melacak individu di seluruh frame menggunakan ID unik dan menghitung metrik fokus.
- **Dashboard Interaktif**: Antarmuka berbasis Streamlit untuk memvisualisasikan data dan status secara real-time.
- **Visual Ramah Pengguna**: Menampilkan bounding box dengan status fokus pada wajah yang terdeteksi.

---

## 📋 Persyaratan

### Library Python:

- `opencv-python`
- `mediapipe`
- `ultralytics`
- `pandas`
- `streamlit`
- `math`
- `time`

### Model:
- File bobot YOLOv8 (`yolov8l.pt`) harus diunduh dan ditempatkan di direktori `model/`.

---

## 🚀 Instalasi

1. Instal dependensi:
   ```bash
   pip install -r requirements.txt
   ```

2. Tempatkan file model YOLOv8:
   - Unduh `yolov8l.pt` dari [Ultralytics YOLOv8 repository](https://github.com/ultralytics/ultralytics) atau file model YOLOv8 yang sudah dilatih.
   - Simpan di direktori `model/`.

3. Jalankan aplikasi:
   ```bash
   streamlit run app.py
   ```

---

## 🛠️ Cara Kerja

1. **Deteksi Wajah**: Model YOLOv8 mendeteksi wajah di setiap frame dari umpan kamera.
2. **Penentuan Fokus**: MediaPipe menganalisis arah pandangan untuk menentukan apakah seseorang fokus berdasarkan posisi mata.
3. **Pelacakan Data**: Setiap wajah yang terdeteksi diberi ID unik, dan metrik fokus (durasi fokus dan tidak fokus) dihitung.
4. **Visualisasi**: Bounding box dan status ditampilkan secara real-time pada video feed, bersama dengan ringkasan fokus dalam format tabel.

---

## 🎥 Antarmuka Streamlit

- **Video Feed**: Menampilkan umpan kamera real-time dengan bounding box dan indikator status (Focused/Not Focused).
- **Tabel Ringkasan fokus**: Menampilkan detail untuk setiap individu:
  - ID Unik
  - Status Fokus Saat Ini
  - Durasi Fokus (detik)
  - Durasi Tidak Fokus (detik)

---

## ⚙️ Kustomisasi

- **Ganti Model**: Ganti `yolov8l.pt` dengan model YOLOv8 Anda sendiri untuk deteksi khusus.
- **Ambang Fokus**: Sesuaikan fungsi `is_focused` untuk mengubah parameter analisis pandangan.
- **Kepercayaan Deteksi**: Ubah `conf=0.5` pada panggilan prediksi YOLO untuk mengatur ambang kepercayaan.


  


 
