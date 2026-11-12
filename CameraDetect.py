"""
CameraDetect.py
===============
Mode Foto dari Kamera:
  - Tampilkan live preview webcam
  - Tekan SPASI atau klik tombol [Ambil Foto] untuk capture
  - Sistem langsung mendeteksi golongan kendaraan dari foto tersebut
  - Hasil foto tersimpan di folder Hasil/
  - Dapat capture berkali-kali tanpa menutup jendela
"""

import os
import datetime
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Re-use fungsi dari Track.py
# ---------------------------------------------------------------------------
from Track import (
    get_detector_model,
    get_classifier_model,
    draw_vehicle_boxes,
    normalize_golongan_name,
    CLASSES,
    GOLONGAN_COLORS,
    HASIL_DIR,
)

# ---------------------------------------------------------------------------
# Konstanta UI (warna dalam format hex)
# ---------------------------------------------------------------------------
BG_DARK   = "#0d0f1a"
BG_PANEL  = "#12162a"
BG_CARD   = "#1a1f35"
ACCENT    = "#00d4ff"
ACCENT2   = "#7c3aed"
SUCCESS   = "#22c55e"
WARNING   = "#f59e0b"
DANGER    = "#ef4444"
TEXT_PRI  = "#f0f4ff"
TEXT_SEC  = "#8892b0"
BORDER    = "#2a3050"

GOLONGAN_HEX = {
    "Golongan I":   "#22c55e",
    "Golongan II":  "#f97316",
    "Golongan III": "#eab308",
    "Golongan IV":  "#06b6d4",
    "Golongan V":   "#ef4444",
    "Golongan VI":  "#a855f7",
}

GOLONGAN_DESC = {
    "Golongan I":   "Sedan / Jip / Pick Up / Bus / Mobil Pribadi",
    "Golongan II":  "Truk Besar 2 Gandar",
    "Golongan III": "Truk Besar 3 Gandar",
    "Golongan IV":  "Truk Besar 4 Gandar",
    "Golongan V":   "Truk Besar 5 Gandar atau Lebih",
    "Golongan VI":  "Sepeda Motor / Roda 2",
}


# ---------------------------------------------------------------------------
# Aplikasi Utama
# ---------------------------------------------------------------------------
class CameraDetectApp:
    """GUI Capture Foto dari Kamera + Deteksi Golongan Kendaraan."""

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
        self.detector = None
        self.classifier = None
        self.running = False
        self.last_frame = None          # Frame mentah terakhir dari kamera
        self.result_frame = None        # Frame hasil deteksi setelah capture
        self.capture_lock = threading.Lock()
        self.capture_requested = False

        self._build_gui()
        self._load_models()
        self._start_camera()

    # ------------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------------
    def _build_gui(self):
        self.root = tk.Tk()
        self.root.title("Deteksi Golongan Kendaraan via Foto Kamera")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(True, True)

        # --- Judul ---
        title_frame = tk.Frame(self.root, bg=BG_DARK)
        title_frame.pack(fill=tk.X, padx=20, pady=(18, 5))

        tk.Label(
            title_frame,
            text="Deteksi Golongan Kendaraan",
            font=("Segoe UI", 17, "bold"),
            bg=BG_DARK, fg=ACCENT,
        ).pack(side=tk.LEFT)

        self.lbl_mode = tk.Label(
            title_frame,
            text="Memuat model...",
            font=("Segoe UI", 9),
            bg=BG_DARK, fg=TEXT_SEC,
        )
        self.lbl_mode.pack(side=tk.RIGHT, padx=4)

        # --- Area Utama (preview kiri | hasil kanan) ---
        main_frame = tk.Frame(self.root, bg=BG_DARK)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)

        # Kiri — Preview Kamera
        left = tk.Frame(main_frame, bg=BG_CARD, relief=tk.FLAT, bd=0)
        left.grid(row=0, column=0, padx=(0, 8), pady=0, sticky="nsew")
        main_frame.columnconfigure(0, weight=3)

        tk.Label(left, text="Preview Kamera", font=("Segoe UI", 10, "bold"),
                 bg=BG_CARD, fg=TEXT_SEC).pack(anchor=tk.W, padx=12, pady=(10, 4))

        self.canvas_preview = tk.Canvas(
            left, width=480, height=360,
            bg="#080a14", highlightthickness=1, highlightbackground=BORDER,
        )
        self.canvas_preview.pack(padx=12, pady=(0, 12))
        self.canvas_preview.create_text(
            240, 180, text="Menginisialisasi kamera...",
            fill=TEXT_SEC, font=("Segoe UI", 11), tags="placeholder"
        )

        # Kanan — Hasil Deteksi
        right = tk.Frame(main_frame, bg=BG_CARD, relief=tk.FLAT, bd=0)
        right.grid(row=0, column=1, padx=(8, 0), pady=0, sticky="nsew")
        main_frame.columnconfigure(1, weight=3)

        tk.Label(right, text="Hasil Deteksi", font=("Segoe UI", 10, "bold"),
                 bg=BG_CARD, fg=TEXT_SEC).pack(anchor=tk.W, padx=12, pady=(10, 4))

        self.canvas_result = tk.Canvas(
            right, width=480, height=360,
            bg="#080a14", highlightthickness=1, highlightbackground=BORDER,
        )
        self.canvas_result.pack(padx=12, pady=(0, 12))
        self.canvas_result.create_text(
            240, 180, text="Belum ada foto yang diambil",
            fill=TEXT_SEC, font=("Segoe UI", 11), tags="placeholder"
        )

        # --- Panel Info Golongan (kanan bawah) ---
        info_outer = tk.Frame(right, bg=BG_CARD)
        info_outer.pack(fill=tk.X, padx=12, pady=(0, 10))

        tk.Label(info_outer, text="Hasil Klasifikasi Golongan:",
                 font=("Segoe UI", 9, "bold"), bg=BG_CARD, fg=TEXT_SEC).pack(anchor=tk.W)

        self.frame_stats = tk.Frame(info_outer, bg=BG_CARD)
        self.frame_stats.pack(fill=tk.X, pady=(6, 0))

        self.stat_labels = {}
        for i, gol in enumerate(CLASSES):
            col_frame = tk.Frame(self.frame_stats, bg=BG_PANEL,
                                 padx=10, pady=6, relief=tk.FLAT)
            col_frame.grid(row=0, column=i, padx=3, sticky="ew")
            self.frame_stats.columnconfigure(i, weight=1)

            hex_col = GOLONGAN_HEX.get(gol, "#ffffff")
            tk.Label(col_frame, text=gol.replace("Golongan ", "Gol "),
                     font=("Segoe UI", 8, "bold"), bg=BG_PANEL, fg=hex_col).pack()
            lbl_cnt = tk.Label(col_frame, text="0",
                               font=("Segoe UI", 15, "bold"), bg=BG_PANEL, fg=TEXT_PRI)
            lbl_cnt.pack()
            self.stat_labels[gol] = lbl_cnt

        # --- Rincian Golongan Terbanyak ---
        self.lbl_result_detail = tk.Label(
            right,
            text="-",
            font=("Segoe UI", 11, "bold"),
            bg=BG_CARD, fg=ACCENT,
            wraplength=460,
            justify=tk.CENTER,
        )
        self.lbl_result_detail.pack(pady=(2, 10))

        # --- Panel Kontrol ---
        ctrl_frame = tk.Frame(self.root, bg=BG_DARK)
        ctrl_frame.pack(fill=tk.X, padx=20, pady=(5, 18))

        # Tombol Ambil Foto
        self.btn_capture = tk.Button(
            ctrl_frame,
            text="Ambil & Deteksi Foto",
            font=("Segoe UI", 13, "bold"),
            bg=ACCENT2, fg="white",
            activebackground="#5b21b6",
            activeforeground="white",
            relief=tk.FLAT,
            padx=28, pady=10,
            cursor="hand2",
            command=self.request_capture,
            state=tk.DISABLED,
        )
        self.btn_capture.pack(side=tk.LEFT, padx=(0, 10))

        # Hint keyboard
        tk.Label(
            ctrl_frame,
            text="Atau tekan SPASI untuk ambil foto",
            font=("Segoe UI", 9),
            bg=BG_DARK, fg=TEXT_SEC,
        ).pack(side=tk.LEFT)

        # Tombol Simpan
        self.btn_save = tk.Button(
            ctrl_frame,
            text="Simpan Hasil",
            font=("Segoe UI", 10),
            bg=BG_PANEL, fg=TEXT_SEC,
            activebackground=BG_CARD,
            relief=tk.FLAT,
            padx=16, pady=10,
            cursor="hand2",
            command=self._save_result,
            state=tk.DISABLED,
        )
        self.btn_save.pack(side=tk.RIGHT, padx=(10, 0))

        # Tombol Tutup
        tk.Button(
            ctrl_frame,
            text="Tutup",
            font=("Segoe UI", 10),
            bg=BG_PANEL, fg=DANGER,
            activebackground=BG_CARD,
            relief=tk.FLAT,
            padx=16, pady=10,
            cursor="hand2",
            command=self._on_close,
        ).pack(side=tk.RIGHT, padx=(10, 0))

        # --- Status Bar ---
        self.lbl_status = tk.Label(
            self.root,
            text="Memuat model AI, harap tunggu...",
            font=("Segoe UI", 9),
            bg="#0a0d1a", fg=TEXT_SEC,
            anchor=tk.W,
            padx=20, pady=5,
        )
        self.lbl_status.pack(fill=tk.X, side=tk.BOTTOM)

        # Keyboard binding
        self.root.bind("<space>", lambda e: self.request_capture())
        self.root.bind("<Return>", lambda e: self.request_capture())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Pastikan ukuran window cukup
        self.root.update_idletasks()
        self.root.minsize(1040, 640)

    # ------------------------------------------------------------------
    # Model Loading
    # ------------------------------------------------------------------
    def _load_models(self):
        def _load():
            self._set_status("Memuat model detektor default YOLOv8...")
            try:
                self.detector = get_detector_model()
            except Exception as e:
                self._set_status(f"[!] Gagal memuat detector: {e}")
                return

            self.classifier, _ = get_classifier_model()

            # Periksa apakah detektor memiliki kelas tol Golongan I - VI
            has_tol = any("golongan" in str(v).lower() for v in getattr(self.detector, 'names', {}).values())

            if has_tol:
                mode_txt = "Model YOLOv8 Terlatih (Golongan I - VI)"
                mode_color = SUCCESS
            else:
                mode_txt = "Mode Default YOLOv8 (COCO Mode)"
                mode_color = ACCENT

            self.root.after(0, lambda: self.lbl_mode.config(text=mode_txt, fg=mode_color))
            self.root.after(0, lambda: self.btn_capture.config(state=tk.NORMAL))
            self._set_status("Model siap. Tekan SPASI atau [Ambil & Deteksi Foto] untuk capture.")

        threading.Thread(target=_load, daemon=True).start()

    # ------------------------------------------------------------------
    # Kamera
    # ------------------------------------------------------------------
    def _start_camera(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self._set_status("[!] Kamera tidak dapat dibuka. Pastikan kamera tersambung.")
            return
        self.running = True
        threading.Thread(target=self._camera_loop, daemon=True).start()

    def _camera_loop(self):
        """Loop background: baca frame kamera, update preview, handle capture."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            with self.capture_lock:
                self.last_frame = frame.copy()

            # Jika ada permintaan capture, proses sekarang
            if self.capture_requested and self.detector is not None:
                self.capture_requested = False
                self._process_capture(frame.copy())

            # Tampilkan preview (resize agar sesuai canvas)
            preview = cv2.resize(frame, (480, 360))
            # Gambar garis panduan tengah
            h, w = preview.shape[:2]
            cx, cy = w // 2, h // 2
            cv2.line(preview, (cx - 20, cy), (cx + 20, cy), (0, 200, 255), 1)
            cv2.line(preview, (cx, cy - 20), (cx, cy + 20), (0, 200, 255), 1)
            cv2.rectangle(preview, (cx - 120, cy - 90), (cx + 120, cy + 90), (0, 200, 255), 1)

            # Convert ke PhotoImage dan tampilkan di canvas
            rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            self._update_canvas(self.canvas_preview, rgb)

        if self.cap:
            self.cap.release()

    def _update_canvas(self, canvas, rgb_array):
        """Update tkinter canvas dari numpy RGB array (thread-safe via after)."""
        from PIL import Image, ImageTk
        img = Image.fromarray(rgb_array)
        photo = ImageTk.PhotoImage(image=img)

        def _draw():
            canvas.delete("all")
            canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            canvas._photo_ref = photo  # Simpan referensi agar tidak di-GC

        self.root.after(0, _draw)

    # ------------------------------------------------------------------
    # Capture & Deteksi
    # ------------------------------------------------------------------
    def request_capture(self):
        """Dipanggil saat user menekan tombol atau SPASI."""
        if self.detector is None:
            self._set_status("[!] Model belum selesai dimuat, harap tunggu.")
            return
        self.capture_requested = True
        self._set_status("Mengambil foto & menjalankan deteksi...")
        self.btn_capture.config(state=tk.DISABLED, text="Mendeteksi...")

    def _process_capture(self, frame):
        """Proses deteksi pada frame yang di-capture (berjalan di thread kamera)."""
        try:
            result = draw_vehicle_boxes(self.detector, self.classifier, frame.copy(), conf=0.15)
            self.result_frame = result.copy()

            # Kumpulkan statistik dari hasil overlay
            stats = self._get_stats(frame)

            # Update UI di main thread
            self.root.after(0, lambda: self._update_result_ui(result, stats))
        except Exception as e:
            self.root.after(0, lambda: self._set_status(f"[!] Error saat deteksi: {e}"))
            self.root.after(0, lambda: self.btn_capture.config(
                state=tk.NORMAL, text="Ambil & Deteksi Foto"))

    def _get_stats(self, frame):
        """Jalankan deteksi untuk mendapat statistik golongan (mendukung model custom dan COCO)."""
        from Track import COCO_CLASS_MAPPING
        stats = {g: 0 for g in CLASSES}
        has_tol = any(
            ("golongan" in str(v).lower() or str(v).lower().startswith("gol"))
            for v in getattr(self.detector, 'names', {}).values()
        )
        try:
            results = self.detector.predict(frame, conf=0.15, verbose=False)
            if results and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    raw_name = str(self.detector.names.get(cls_id, "")).strip()
                    raw_lower = raw_name.lower()

                    # 1. Cek jika model custom (Golongan I - VI atau GOL I - V)
                    golongan_name = normalize_golongan_name(raw_name, cls_id if has_tol else None)

                    # 2. Fallback untuk model default COCO (hanya jika BUKAN model custom)
                    if not has_tol and golongan_name is None:
                        if raw_lower in COCO_CLASS_MAPPING:
                            golongan_name = COCO_CLASS_MAPPING[raw_lower]
                        elif raw_lower in ["vehicle", "automobile"]:
                            golongan_name = "Golongan I"

                    if golongan_name and golongan_name in CLASSES:
                        stats[golongan_name] = stats.get(golongan_name, 0) + 1
        except Exception:
            pass
        return stats

    def _update_result_ui(self, result_frame, stats):
        """Update tampilan hasil deteksi di main thread."""
        # Tampilkan gambar hasil di canvas kanan
        resized = cv2.resize(result_frame, (480, 360))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self._update_canvas(self.canvas_result, rgb)

        # Update kartu statistik
        total = sum(stats.values())
        for gol, lbl in self.stat_labels.items():
            lbl.config(text=str(stats.get(gol, 0)))

        # Cari golongan terbanyak
        if total > 0:
            top_gol = max(stats, key=stats.get)
            top_cnt = stats[top_gol]
            desc    = GOLONGAN_DESC.get(top_gol, "")
            hex_col = GOLONGAN_HEX.get(top_gol, ACCENT)
            detail_txt = f"{top_gol} ({top_cnt} kendaraan) - {desc}"
            self.lbl_result_detail.config(text=detail_txt, fg=hex_col)
            self._set_status(
                f"Ditemukan {total} kendaraan. "
                f"Terbanyak: {top_gol} ({top_cnt} unit) - {desc}"
            )
        else:
            self.lbl_result_detail.config(text="Tidak ada kendaraan terdeteksi dalam foto ini.", fg=WARNING)
            self._set_status("Tidak ada kendaraan terdeteksi. Coba arahkan kamera ke kendaraan.")

        # Auto-save
        self._auto_save(result_frame)

        self.btn_capture.config(state=tk.NORMAL, text="Ambil & Deteksi Foto")
        self.btn_save.config(state=tk.NORMAL)

    # ------------------------------------------------------------------
    # Simpan Hasil
    # ------------------------------------------------------------------
    def _auto_save(self, frame):
        """Simpan hasil deteksi otomatis ke folder Hasil/."""
        try:
            os.makedirs(HASIL_DIR, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(HASIL_DIR, f"capture_{ts}.jpg")
            cv2.imwrite(out_path, frame)
            print(f"[+] Auto-save: {out_path}")
        except Exception as e:
            print(f"[!] Auto-save gagal: {e}")

    def _save_result(self):
        """Simpan ulang hasil (manual save dengan dialog)."""
        if self.result_frame is None:
            return
        from tkinter import filedialog
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"deteksi_golongan_{ts}.jpg"
        path = filedialog.asksaveasfilename(
            title="Simpan Hasil Deteksi",
            initialfile=default_name,
            defaultextension=".jpg",
            filetypes=[("JPEG Image", "*.jpg"), ("PNG Image", "*.png"), ("All Files", "*.*")],
        )
        if path:
            cv2.imwrite(path, self.result_frame)
            self._set_status(f"[+] Berhasil disimpan: {path}")
            messagebox.showinfo("Simpan Berhasil", f"Hasil deteksi disimpan ke:\n{path}")

    # ------------------------------------------------------------------
    # Utilitas
    # ------------------------------------------------------------------
    def _set_status(self, msg: str):
        self.root.after(0, lambda: self.lbl_status.config(text=msg))

    def _on_close(self):
        self.running = False
        self.root.after(200, self.root.destroy)

    def run(self):
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Entry Point Standalone
# ---------------------------------------------------------------------------
def launch_camera_detect(camera_index: int = 0):
    """Buka jendela Deteksi Kendaraan via Foto Kamera."""
    try:
        import PIL
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])

    app = CameraDetectApp(camera_index=camera_index)
    app.run()


if __name__ == "__main__":
    launch_camera_detect()
