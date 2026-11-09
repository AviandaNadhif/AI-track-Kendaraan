# AI-track-Kendaraan

## Bisa memilih memakai 
1. COCO default
2. Mode Custom (golongan VI COCO default)


Aplikasi **deteksi, klasifikasi, dan tracking kendaraan** berstandar **Golongan I - VI Jalan Tol Indonesia** berbasis **YOLOv8 Object Detection** dari Ultralytics.

---

## Skema Golongan Kendaraan Jalan Tol Indonesia

Klasifikasi mengacu pada regulasi penetapan tarif golongan kendaraan jalan tol di Indonesia:

| Golongan | Jenis Kendaraan | Warna Bounding Box (BGR) | Kode Hex |
| :--- | :--- | :--- | :--- |
| **Golongan I** | Sedan, Jip, Pick Up, Bus, Mobil Pribadi | 🟢 Hijau `(0, 255, 0)` | `#22c55e` |
| **Golongan II** | Truk Besar (2 Gandar) | 🟠 Oranye `(255, 128, 0)` | `#f97316` |
| **Golongan III** | Truk Besar (3 Gandar / Tronton) | 🔵 Cyan `(255, 255, 0)` | `#eab308` |
| **Golongan IV** | Truk Besar (4 Gandar / Trintin / Trinton) | 🟡 Kuning `(0, 255, 255)` | `#06b6d4` |
| **Golongan V** | Truk Besar (5 Gandar atau Lebih / Multi-Axle) | 🔶 Merah-Oranye `(0, 69, 255)` | `#ef4444` |
| **Golongan VI** | Kendaraan Roda 2 (Sepeda Motor / Skuter) | 🟣 Magenta `(255, 0, 255)` | `#a855f7` |

---

## Fitur

1. **Smart Fallback Golongan VI ke Pretrained COCO**:
   - Memecahkan kendala bounding box sepeda motor pada model tol lokal dengan memanfaatkan model pretrained COCO (`yolov8s.pt` / `yolov8n.pt`).
   - Menghasilkan bounding box Golongan VI yang presisi, proporsional, dan rapi tanpa terputus ketika ada pengendara di atas sepeda motor.
   - Pada mode Model Custom (`runs/best.pt`), fallback COCO aktif secara otomatis untuk melengkapi atau memperbaiki deteksi Golongan VI yang terlewat.

2. **Multi-Vehicle Scanning Presisi Tinggi**:
   - Default threshold confidence diatur ke **`0.15`** (seperti pada versi 4), sehingga sistem mampu mendeteksi kendaraan di kejauhan, kendaraan kecil, serta rombongan konvoi secara simultan.
   - Menonaktifkan `agnostic_nms=True` pada detektor agar kendaraan berbeda kelas yang berdampingan atau beriringan di lajur tol tidak saling meniadakan.
   - Menggunakan mekanisme penyaringan duplikasi fleksibel (`overlap_threshold = 0.60`) untuk mempertahankan banyak kendaraan di jalan raya.

3. **Pilihan Model AI Fleksibel di GUI**:
   - **Model Standar YOLO (yolov8s.pt / yolov8n.pt - Pretrained COCO)**: Deteksi multi-kendaraan dengan kecepatan tinggi dan akurasi tinggi untuk lalu lintas umum.
   - **Model Custom Tol (runs/best.pt + COCO Fallback Golongan VI)**: Model hasil transfer learning mandiri spesialis truk multi-gandar jalan tol (Golongan I–V), dengan integrasi fallback COCO untuk Golongan VI.

4. **Modul Pengujian Terstandarisasi Jurnal (`evaluate_golongan.py`)**:
   - Dirancang untuk pengujian 10 citra uji per golongan (`Dataset_Uji/`, total 60 citra uji independen).
   - Menghitung akurasi klasifikasi, *Confusion Matrix*, interval kepercayaan *Wilson Score* (95%), serta pengukuran latensi inferensi (FPS/ms) pada CPU dan GPU NVIDIA.
   - Otomatis mengekspor data ke format CSV (`log_pengujian.csv`) dan JSON (`hasil_uji.json`) di folder `Hasil_Uji/` agar tabel naskah jurnal dapat diisi secara otomatis.

5. **Pemeriksaan Kebocoran Data / Data Leakage (`cek_kebocoran.py`)**:
   - Memeriksa integritas dataset uji terhadap dataset latih dan validasi menggunakan kombinasi **dHash (Difference Hash)** dan **MD5 Cryptographic Hash**.
   - Menjamin bahwa tidak ada citra uji yang identik atau duplikat dengan citra latih guna memenuhi standar metodologi penulisan jurnal ilmiah.

6. **Modul Foto Kamera Terpisah (`CameraDetect.py`)**:
   - Antarmuka split-view untuk preview webcam live (kiri) berdampingan dengan hasil capture dan anotasi deteksi (kanan).
   - Menampilkan visualisasi kartu statistik jumlah kendaraan per golongan dan penentuan golongan dominan.

---

## Struktur Proyek `AI_Track_kendaraan`

```
AI_Track_kendaraan_v6 (Sesuai Jurnal)/
├── Track.py              # Aplikasi utama (GUI 4 Mode + Multi-Vehicle Tracking + Fallback COCO)
├── CameraDetect.py       # GUI terpisah: live webcam capture, anotasi foto, & kartu statistik
├── evaluate_golongan.py  # Skrip evaluasi formal untuk naskah jurnal (Akurasi, Wilson Score, CSV/JSON)
├── cek_kebocoran.py      # Verifikasi data leakage (dHash & MD5) antara Dataset_Uji vs Train/Valid
├── train.py              # GUI training custom model YOLOv8 (NVIDIA Tensor Core Ready)
├── data.yaml             # Konfigurasi dataset YOLO (path, class names Golongan I - VI)
├── yolov8s.pt            # Bobot model YOLOv8
├── README.md             # Dokumentasi teknis proyek ini
├── Dataset/              # Dataset kustom per golongan (Total 551 gambar)
│   ├── Golongan I/       # Sedan, Bus, Pick Up, Mobil Pribadi
│   ├── Golongan II/      # Truk 2 Gandar
│   ├── Golongan III/     # Truk 3 Gandar / Tronton
│   ├── Golongan IV/      # Truk 4 Gandar / Trintin / Trinton
│   ├── Golongan V/       # Truk 5 Gandar / Multi-Axle
│   └── Golongan VI/      # Sepeda Motor
├── Dataset_Original/     # Backup dataset asli Roboflow (train: 1272, valid: 121)
├── Dataset_Uji/          # Dataset uji formal untuk naskah jurnal (10 citra per golongan = 60 citra)
│   ├── Golongan I/
│   ├── Golongan II/
│   ├── Golongan III/
│   ├── Golongan IV/
│   ├── Golongan V/
│   └── Golongan VI/
├── train/                # Dataset training aktif (1639 images + 1639 labels)
│   ├── images/
│   └── labels/
├── valid/                # Dataset validasi aktif (215 images + 215 labels)
│   ├── images/
│   └── labels/
├── test/                 # Dataset test aktif (120 images + 120 labels)
│   ├── images/
│   └── labels/
├── Hasil/                # Output penyimpanan foto & video beranotasi
├── Hasil_Uji/            # Output laporan pengujian jurnal (log_pengujian.csv, hasil_uji.json, ringkasan.txt)
└── runs/                 # Output checkpoint Ultralytics (weights/best.pt & weights/last.pt)
```

---

## Requirements & Instalasi

Pastikan menggunakan Python 3.10 ke atas. Pasang dependensi:

```powershell
pip install ultralytics opencv-python numpy splitfolders pillow torch
```

> **Akselerasi GPU NVIDIA (CUDA):**
> ```powershell
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
> ```

---

## Cara Menjalankan Aplikasi

### 1. Aplikasi Utama (GUI 4 Mode):
```powershell
python Track.py
```

### 2. Evaluasi Ilmiah untuk Naskah Jurnal:
```powershell
# Pengujian dataset uji (kecepatan & akurasi)
python evaluate_golongan.py

# Pengujian dengan metrik validasi Ultralytics (mAP, Precision, Recall)
python evaluate_golongan.py --val

# Menjalankan evaluasi pada GPU (device 0) atau CPU
python evaluate_golongan.py --device 0
python evaluate_golongan.py --device cpu
```

### 3. Pemeriksaan Kebocoran Data (Data Leakage):
```powershell
python cek_kebocoran.py
```

### 4. Mode Capture Foto Kamera (GUI Terpisah):
```powershell
python CameraDetect.py
```

### 5. Training Model Custom:
```powershell
python train.py
```

---

## Panduan Antarmuka GUI Utama — `Track.py`

Jendela utama menyediakan pilihan model AI dan 4 tombol mode input:

### Pilihan Model AI:
- **Model Standar YOLO (yolov8s.pt - Pretrained COCO Multi-Vehicle)**: Menggunakan pemetaan COCO penuh, sangat responsif mendeteksi banyak kendaraan bersamaan (mobil, bus, truk 2 gandar, motor).
- **Model Custom Tol (runs/best.pt + Fallback COCO Golongan VI)**: Mengutamakan model lokal hasil training untuk mengenali truk golongan II–V, dilengkapi fallback otomatis ke COCO untuk Golongan VI dan kendaraan konvoi.

### Tombol Aksi Input:
| Tombol | Fungsi & Kegunaan |
| :--- | :--- |
| 📷 **Kamera Live** | Tracking kendaraan real-time via webcam/kamera CCTV jalan tol dengan header statistik real-time. |
| 🖼️ **Upload Foto** | Memilih satu atau beberapa foto kendaraan untuk dideteksi dan otomatis disimpan ke folder `Hasil/`. |
| 📁 **Upload Video** | Memproses rekaman video lalu lintas secara frame-by-frame dan menyimpan video beranotasi ke folder `Hasil/`. |
| 🏋️ **Custom Train** | Membuka jendela antarmuka training model (`train.py`) dengan akselerasi NVIDIA Tensor Core. |

---

## 🔄 Logika Smart Fallback & Deteksi Multi-Kendaraan

```
                         Input Frame (Citra / Video)
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
         [Model Custom runs/best.pt]         [Model Standar COCO]
         Deteksi Tol Golongan I - V           Deteksi Objek COCO
                    │                                 │
     (Cek Golongan VI & Missing Box)                  │
                    │                                 │
                    ▼                                 ▼
         [Fallback COCO Pretrained]           Pemetaan Kelas:
         - motorcycle/bicycle -> Gol VI       - car, bus   -> Gol I
         - Lengkapi box konvoi terlewat       - truck      -> Gol II
                    │                         - motorcycle -> Gol VI
                    └────────────────┬────────────────┘
                                     ▼
                    [Collision Filter: IoU < 0.60]
                                     ▼
               Output Visual: Bounding Box & Statistik Golongan
```

### Pemetaan COCO ke Golongan Tol:
- `car`, `bus` ➔ **Golongan I**
- `truck` ➔ **Golongan II**
- `motorcycle`, `bicycle` ➔ **Golongan VI**

---

## 📊 Metrik Evaluasi Naskah Jurnal (`Hasil_Uji/`)

Setiap kali `evaluate_golongan.py` dijalankan, sistem menghasilkan:
1. `log_pengujian.csv`: Rincian pengujian per gambar (Ground Truth, Prediksi, Status Benar/Salah, Nilai Confidence, Waktu Inferensi).
2. `hasil_uji.json`: Ringkasan metrik statistik (Akurasi total, Akurasi per golongan, Latensi rata-rata ms & FPS, Interval Kepercayaan Wilson 95%).
3. `ringkasan.txt`: Ringkasan teks formal siap pakai untuk disalin ke bab Hasil dan Pembahasan naskah jurnal.
