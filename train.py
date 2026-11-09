import os
import math
import random
import shutil
import threading
import subprocess
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import splitfolders
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Golongan Kendaraan di Indonesia (Golongan I - VI)
CLASSES = [
    "Golongan I",
    "Golongan II",
    "Golongan III",
    "Golongan IV",
    "Golongan V",
    "Golongan VI"
]

def prepare_classification_dataset(raw_dataset_dir, log_callback=print):
    """
    Membagi dataset folder (Dataset/Golongan I .. Golongan VI)
    menjadi Dataset_Split (80% Train, 20% Val) untuk YOLO Classification Training.
    """
    out_dir = os.path.join(os.path.dirname(__file__), "Dataset_Split")

    if os.path.exists(out_dir):
        try:
            shutil.rmtree(out_dir)
        except Exception as e:
            log_callback(f"[!] Warning saat membersihkan Dataset_Split: {e}")

    log_callback("[+] Membagi dataset dengan split-folders (80% Train, 20% Val)...")
    splitfolders.ratio(raw_dataset_dir, output=out_dir, seed=42, ratio=(0.8, 0.2))

    # Pastikan setiap subfolder Golongan I - VI ada di train dan val
    for split in ["train", "val"]:
        split_path = os.path.join(out_dir, split)
        for cls_name in CLASSES:
            os.makedirs(os.path.join(split_path, cls_name), exist_ok=True)

    # Hitung statistik gambar
    total_imgs = 0
    for root, _, files in os.walk(out_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                total_imgs += 1

    log_callback(f"[+] Dataset siap: {total_imgs} gambar terbagi di {out_dir}")
    return out_dir

GOLONGAN_MAP = {
    "Golongan I": 0,
    "Golongan II": 1,
    "Golongan III": 2,
    "Golongan IV": 3,
    "Golongan V": 4,
    "Golongan VI": 5,
}

# ═══════════════════════════════════════════════════════════════════════
#   AUTO-RECOVERY: Rebuild train/ dan valid/ jika hilang
# ═══════════════════════════════════════════════════════════════════════

DATA_YAML_TEMPLATE = """path: {root_path}
train: ../train/images
val: ../valid/images
test: ../test/images

names:
  0: Golongan I
  1: Golongan II
  2: Golongan III
  3: Golongan IV
  4: Golongan V
  5: Golongan VI
"""

def generate_data_yaml(root_dir, log_callback=print):
    """Regenerasi file data.yaml jika hilang."""
    yaml_path = os.path.join(root_dir, "data.yaml")
    root_unix = root_dir.replace("\\", "/")
    content = DATA_YAML_TEMPLATE.format(root_path=root_unix).strip() + "\n"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)
    log_callback(f"[+] data.yaml berhasil di-regenerasi di: {yaml_path}")
    return yaml_path


def _is_dataset_intact(root_dir):
    """Cek apakah train/, valid/, dan test/ masih lengkap (images + labels ada)."""
    train_imgs = os.path.join(root_dir, "train", "images")
    train_lbls = os.path.join(root_dir, "train", "labels")
    valid_imgs = os.path.join(root_dir, "valid", "images")
    valid_lbls = os.path.join(root_dir, "valid", "labels")
    test_imgs = os.path.join(root_dir, "test", "images")
    test_lbls = os.path.join(root_dir, "test", "labels")

    for d in [train_imgs, train_lbls, valid_imgs, valid_lbls, test_imgs, test_lbls]:
        if not os.path.exists(d):
            return False
        # Folder ada tapi kosong = tidak intact
        if len(os.listdir(d)) == 0:
            return False
    return True


def _copy_image_robust(src_path, dst_path):
    """Salin citra ke tujuan, konversi ke JPG jika format AVIF untuk kompatibilitas penuh."""
    ext = os.path.splitext(src_path)[1].lower()
    if ext == ".avif":
        dst_path_jpg = os.path.splitext(dst_path)[0] + ".jpg"
        try:
            from PIL import Image
            with Image.open(src_path) as pil_img:
                pil_img.convert("RGB").save(dst_path_jpg, "JPEG", quality=95)
            return dst_path_jpg
        except Exception:
            shutil.copy2(src_path, dst_path)
            return dst_path
    else:
        shutil.copy2(src_path, dst_path)
        return dst_path


def rebuild_dataset_from_sources(log_callback=print):
    """
    Rebuild seluruh dataset YOLO (train/, valid/, dan test/) dari tiga sumber:
      1. Dataset_Original/ — backup data Roboflow (images + labels dengan bbox akurat)
      2. Dataset/          — gambar custom per Golongan (80% train, 20% val)
      3. Dataset_Uji/      — gambar uji per Golongan (100% test)

    Juga regenerasi data.yaml jika hilang.
    Dipanggil otomatis saat train/, valid/, atau test/ tidak ditemukan atau kosong.
    """
    root_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(root_dir, "Dataset")
    dataset_uji_dir = os.path.join(root_dir, "Dataset_Uji")
    original_dir = os.path.join(root_dir, "Dataset_Original")

    log_callback("=" * 60)
    log_callback("[!] RECOVERY MODE: dataset hilang/kosong!")
    log_callback("    Memulai rebuild dataset (train/, valid/, test/) dari sumber...")
    log_callback("=" * 60)

    # Buat ulang folder tujuan
    train_img_dir = os.path.join(root_dir, "train", "images")
    train_lbl_dir = os.path.join(root_dir, "train", "labels")
    valid_img_dir = os.path.join(root_dir, "valid", "images")
    valid_lbl_dir = os.path.join(root_dir, "valid", "labels")
    test_img_dir = os.path.join(root_dir, "test", "images")
    test_lbl_dir = os.path.join(root_dir, "test", "labels")
    for d in [train_img_dir, train_lbl_dir, valid_img_dir, valid_lbl_dir, test_img_dir, test_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.avif')
    total_restored = 0

    # ── SUMBER 1: Dataset_Original (Roboflow backup) ──────────────────
    if os.path.exists(original_dir):
        log_callback("\n[1/3] Mengembalikan data Roboflow dari Dataset_Original/...")
        for split_name, dst_img, dst_lbl in [
            ("train", train_img_dir, train_lbl_dir),
            ("valid", valid_img_dir, valid_lbl_dir),
            ("test", test_img_dir, test_lbl_dir),
        ]:
            src_img = os.path.join(original_dir, split_name, "images")
            src_lbl = os.path.join(original_dir, split_name, "labels")
            if not os.path.exists(src_img):
                continue

            count = 0
            for fname in os.listdir(src_img):
                if fname.lower().endswith(exts):
                    shutil.copy2(os.path.join(src_img, fname), os.path.join(dst_img, fname))
                    # Salin label jika ada
                    lbl_name = os.path.splitext(fname)[0] + ".txt"
                    lbl_src = os.path.join(src_lbl, lbl_name)
                    if os.path.exists(lbl_src):
                        shutil.copy2(lbl_src, os.path.join(dst_lbl, lbl_name))
                    count += 1
            total_restored += count
            log_callback(f"   [{split_name}] {count} gambar Roboflow dipulihkan.")
    else:
        log_callback("\n[1/3] Dataset_Original/ tidak ditemukan — skip data Roboflow.")
        log_callback("      (Training hanya dari gambar custom di Dataset/)")

    # ── SUMBER 2: Dataset/ (gambar custom per Golongan ke train & valid) ─────────
    if os.path.exists(dataset_dir):
        log_callback("\n[2/3] Mendaftarkan gambar custom dari Dataset/...")
        custom_count = 0
        for gol, cls_id in GOLONGAN_MAP.items():
            g_dir = os.path.join(dataset_dir, gol)
            if not os.path.exists(g_dir):
                continue
            clean_gol = gol.lower().replace(" ", "_")
            imgs = [f for f in os.listdir(g_dir) if f.lower().endswith(exts)]

            if not imgs:
                continue

            # Split 80/20 secara acak per golongan
            random.shuffle(imgs)
            split_idx = max(1, math.floor(len(imgs) * 0.8))
            train_imgs_list = imgs[:split_idx]
            val_imgs_list = imgs[split_idx:]

            for img_list, img_dst, lbl_dst in [
                (train_imgs_list, train_img_dir, train_lbl_dir),
                (val_imgs_list, valid_img_dir, valid_lbl_dir),
            ]:
                for f in img_list:
                    target_name = f"custom_{clean_gol}_{f}"
                    src_file = os.path.join(g_dir, f)
                    final_dst = _copy_image_robust(src_file, os.path.join(img_dst, target_name))
                    final_name = os.path.basename(final_dst)
                    lbl_path = os.path.join(lbl_dst, os.path.splitext(final_name)[0] + ".txt")
                    with open(lbl_path, "w", encoding="utf-8") as lf:
                        lf.write(f"{cls_id} 0.500000 0.500000 0.900000 0.900000\n")
                    custom_count += 1

            log_callback(f"   [{gol}] {len(train_imgs_list)} train + {len(val_imgs_list)} val = {len(imgs)} gambar")
        total_restored += custom_count
        log_callback(f"\n   Total gambar custom: {custom_count}")
    else:
        log_callback("\n[2/3] Dataset/ tidak ditemukan — tidak ada gambar custom.")

    # ── SUMBER 3: Dataset_Uji/ (gambar uji custom per Golongan ke test/) ─────────
    if os.path.exists(dataset_uji_dir):
        log_callback("\n[3/3] Mendaftarkan gambar uji dari Dataset_Uji/ ke test/...")
        uji_count = 0
        for gol, cls_id in GOLONGAN_MAP.items():
            g_dir = os.path.join(dataset_uji_dir, gol)
            if not os.path.exists(g_dir):
                continue
            clean_gol = gol.lower().replace(" ", "_")
            imgs = [f for f in os.listdir(g_dir) if f.lower().endswith(exts)]

            if not imgs:
                continue

            for f in imgs:
                target_name = f"uji_{clean_gol}_{f}"
                src_file = os.path.join(g_dir, f)
                final_dst = _copy_image_robust(src_file, os.path.join(test_img_dir, target_name))
                final_name = os.path.basename(final_dst)
                lbl_path = os.path.join(test_lbl_dir, os.path.splitext(final_name)[0] + ".txt")
                with open(lbl_path, "w", encoding="utf-8") as lf:
                    lf.write(f"{cls_id} 0.500000 0.500000 0.900000 0.900000\n")
                uji_count += 1

            log_callback(f"   [{gol}] {len(imgs)} gambar uji dimasukkan ke test/")
        total_restored += uji_count
        log_callback(f"\n   Total gambar uji dari Dataset_Uji/: {uji_count}")
    else:
        log_callback("\n[3/3] Dataset_Uji/ tidak ditemukan — skip gambar uji custom.")

    # ── Regenerasi data.yaml jika hilang ──────────────────────────────
    yaml_path = os.path.join(root_dir, "data.yaml")
    if not os.path.exists(yaml_path):
        generate_data_yaml(root_dir, log_callback)

    # Hapus cache lama agar YOLO tidak pakai data usang
    for cache_file in [
        os.path.join(root_dir, "train", "labels.cache"),
        os.path.join(root_dir, "valid", "labels.cache"),
        os.path.join(root_dir, "test", "labels.cache"),
    ]:
        if os.path.exists(cache_file):
            os.remove(cache_file)
            log_callback(f"   Cache lama dihapus: {os.path.basename(cache_file)}")

    log_callback(f"\n[✓] REBUILD SELESAI! Total {total_restored} gambar siap untuk dataset YOLO.")
    log_callback("=" * 60)
    return total_restored


def check_and_register_dataset_folder(log_callback=print):
    """Sinkronkan gambar yang ada di folder Dataset/ ke dataset YOLO (train/images)."""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(root_dir, "Dataset")
    train_img_dir = os.path.join(root_dir, "train", "images")
    train_lbl_dir = os.path.join(root_dir, "train", "labels")
    val_img_dir = os.path.join(root_dir, "valid", "images")
    if not os.path.exists(dataset_dir) or not os.path.exists(train_img_dir):
        return

    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.avif')
    added = 0
    for gol, cls_id in GOLONGAN_MAP.items():
        g_dir = os.path.join(dataset_dir, gol)
        if not os.path.exists(g_dir):
            continue
        clean_gol = gol.lower().replace(" ", "_")
        for f in os.listdir(g_dir):
            if f.lower().endswith(exts):
                target_base = f"custom_{clean_gol}_{os.path.splitext(f)[0]}"
                already_exists = False
                for d in [train_img_dir, val_img_dir]:
                    if os.path.exists(d):
                        for existing in os.listdir(d):
                            if os.path.splitext(existing)[0] == target_base:
                                already_exists = True
                                break
                    if already_exists:
                        break
                if not already_exists:
                    src_file = os.path.join(g_dir, f)
                    final_dst = _copy_image_robust(src_file, os.path.join(train_img_dir, f"custom_{clean_gol}_{f}"))
                    final_name = os.path.basename(final_dst)
                    lbl_path = os.path.join(train_lbl_dir, os.path.splitext(final_name)[0] + ".txt")
                    with open(lbl_path, "w", encoding="utf-8") as lf:
                        lf.write(f"{cls_id} 0.500000 0.500000 0.900000 0.900000\n")
                    added += 1
    if added > 0:
        log_callback(f"[+] Menambahkan {added} gambar baru dari folder Dataset/ ke training set YOLO.")


def check_and_register_dataset_uji_folder(log_callback=print):
    """Sinkronkan gambar yang ada di folder Dataset_Uji/ ke dataset YOLO (test/images)."""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    uji_dir = os.path.join(root_dir, "Dataset_Uji")
    test_img_dir = os.path.join(root_dir, "test", "images")
    test_lbl_dir = os.path.join(root_dir, "test", "labels")
    if not os.path.exists(uji_dir) or not os.path.exists(test_img_dir):
        return

    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.avif')
    added = 0
    for gol, cls_id in GOLONGAN_MAP.items():
        g_dir = os.path.join(uji_dir, gol)
        if not os.path.exists(g_dir):
            continue
        clean_gol = gol.lower().replace(" ", "_")
        for f in os.listdir(g_dir):
            if f.lower().endswith(exts):
                target_base = f"uji_{clean_gol}_{os.path.splitext(f)[0]}"
                already_exists = False
                for existing in os.listdir(test_img_dir):
                    if os.path.splitext(existing)[0] == target_base:
                        already_exists = True
                        break
                if not already_exists:
                    src_file = os.path.join(g_dir, f)
                    final_dst = _copy_image_robust(src_file, os.path.join(test_img_dir, f"uji_{clean_gol}_{f}"))
                    final_name = os.path.basename(final_dst)
                    lbl_path = os.path.join(test_lbl_dir, os.path.splitext(final_name)[0] + ".txt")
                    with open(lbl_path, "w", encoding="utf-8") as lf:
                        lf.write(f"{cls_id} 0.500000 0.500000 0.900000 0.900000\n")
                    added += 1
    if added > 0:
        log_callback(f"[+] Menambahkan {added} gambar baru dari folder Dataset_Uji/ ke test set YOLO.")

def extract_frames_from_video(video_path, golongan, fps_interval=30, log_callback=print):
    """Ekstrak frame dari video sebagai gambar training untuk Golongan tertentu."""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    class_dir = os.path.join(root_dir, "Dataset", golongan)
    os.makedirs(class_dir, exist_ok=True)

    train_img_dir = os.path.join(root_dir, "train", "images")
    train_lbl_dir = os.path.join(root_dir, "train", "labels")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)

    cls_id = GOLONGAN_MAP.get(golongan, 0)
    clean_gol = golongan.lower().replace(" ", "_")

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    saved = 0
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % fps_interval == 0:
            fname = f"frame_{base_name}_{frame_idx}.jpg"
            out_file = os.path.join(class_dir, fname)
            cv2.imwrite(out_file, frame)

            # Salin ke train YOLO
            train_fname = f"custom_{clean_gol}_{fname}"
            dst_img = os.path.join(train_img_dir, train_fname)
            dst_lbl = os.path.join(train_lbl_dir, os.path.splitext(train_fname)[0] + ".txt")
            cv2.imwrite(dst_img, frame)
            with open(dst_lbl, "w", encoding="utf-8") as lf:
                lf.write(f"{cls_id} 0.500000 0.500000 0.900000 0.900000\n")

            saved += 1
        frame_idx += 1
    cap.release()
    log_callback(f"[+] Diekstrak {saved} frame ke Dataset/{golongan} dan otomatis ditambahkan ke training set.")
    return saved

def add_images_to_dataset(image_paths, golongan, log_callback=print):
    """Salin foto ke Dataset/<Golongan>/ dan otomatis daftarkan ke training set."""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    class_dir = os.path.join(root_dir, "Dataset", golongan)
    os.makedirs(class_dir, exist_ok=True)

    train_img_dir = os.path.join(root_dir, "train", "images")
    train_lbl_dir = os.path.join(root_dir, "train", "labels")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)

    cls_id = GOLONGAN_MAP.get(golongan, 0)
    clean_gol = golongan.lower().replace(" ", "_")

    count = 0
    for img_path in image_paths:
        if os.path.exists(img_path):
            base = os.path.basename(img_path)
            dst = os.path.join(class_dir, base)
            shutil.copy(img_path, dst)

            # Salin ke train YOLO
            train_fname = f"custom_{clean_gol}_{base}"
            dst_img = os.path.join(train_img_dir, train_fname)
            dst_lbl = os.path.join(train_lbl_dir, os.path.splitext(train_fname)[0] + ".txt")
            shutil.copy(img_path, dst_img)
            with open(dst_lbl, "w", encoding="utf-8") as lf:
                lf.write(f"{cls_id} 0.500000 0.500000 0.900000 0.900000\n")

            count += 1
    log_callback(f"[+] {count} foto disalin ke Dataset/{golongan} dan otomatis ditambahkan ke training set.")
    return count

def _balance_dataset_by_oversampling(dataset_split_dir: str, log_callback=print):
    """
    Oversample kelas minoritas agar setiap kelas punya jumlah gambar
    yang sama dengan kelas terbanyak (mencegah bias ke Golongan II).
    """
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    for split in ['train', 'val']:
        split_path = os.path.join(dataset_split_dir, split)
        if not os.path.exists(split_path):
            continue

        # Hitung jumlah gambar tiap kelas
        class_counts = {}
        for cls in os.listdir(split_path):
            cls_path = os.path.join(split_path, cls)
            if os.path.isdir(cls_path):
                imgs = [f for f in os.listdir(cls_path) if f.lower().endswith(exts)]
                class_counts[cls] = imgs

        if not class_counts:
            continue

        max_count = max(len(v) for v in class_counts.values())
        log_callback(f"[+] Oversampling {split}: target {max_count} gambar/kelas")

        for cls, imgs in class_counts.items():
            cls_path = os.path.join(split_path, cls)
            needed = max_count - len(imgs)
            if needed <= 0:
                continue
            # Buat salinan dengan augmentasi ringan (flip horizontal)
            for i in range(needed):
                src = os.path.join(cls_path, imgs[i % len(imgs)])
                img = cv2.imread(src)
                if img is None:
                    continue
                # Alternasi antara flip dan slight brightness shift
                if i % 2 == 0:
                    out = cv2.flip(img, 1)
                else:
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
                    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + np.random.uniform(-25, 25), 0, 255)
                    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
                dst = os.path.join(cls_path, f"os_{i:04d}_{os.path.basename(src)}")
                cv2.imwrite(dst, out)
            log_callback(f"   [{cls}] +{needed} gambar oversampled")


def _print_per_class_accuracy(model_path: str, split_dir: str, log_callback=print):
    """Evaluasi per-kelas dan tampilkan akurasi Golongan II-V secara eksplisit."""
    try:
        log_callback("\n[+] Evaluasi akurasi per golongan...")
        model = YOLO(model_path)
        val_dir = os.path.join(split_dir, 'val')
        if not os.path.exists(val_dir):
            return

        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        correct = {g: 0 for g in CLASSES}
        total   = {g: 0 for g in CLASSES}

        for cls_name in CLASSES:
            cls_path = os.path.join(val_dir, cls_name)
            if not os.path.exists(cls_path):
                continue
            imgs = [f for f in os.listdir(cls_path) if f.lower().endswith(exts)]
            for fname in imgs:
                img_path = os.path.join(cls_path, fname)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                try:
                    res = model.predict(img, verbose=False)
                    if res and hasattr(res[0], 'probs') and res[0].probs is not None:
                        top_idx = int(res[0].probs.top1)
                        pred_name = None
                        if hasattr(model, 'names') and top_idx in model.names:
                            pred_name = str(model.names[top_idx])
                        elif top_idx < len(CLASSES):
                            pred_name = CLASSES[top_idx]
                        total[cls_name] += 1
                        if pred_name == cls_name:
                            correct[cls_name] += 1
                except Exception:
                    pass

        log_callback("\n┌─────────────────────────────────────────┐")
        log_callback("│     AKURASI PER GOLONGAN (Validation)   │")
        log_callback("├─────────────────────────────────────────┤")
        for g in CLASSES:
            t = total[g]
            c = correct[g]
            pct = (c / t * 100) if t > 0 else 0
            bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
            status = '✓' if pct >= 70 else ('⚠' if pct >= 50 else '✗')
            log_callback(f"│ {status} {g:<13} {bar} {pct:5.1f}% ({c}/{t})")
        log_callback("└─────────────────────────────────────────┘")
    except Exception as e:
        log_callback(f"[!] Evaluasi per kelas gagal: {e}")


def get_hardware_acceleration_info():
    """
    Mendeteksi hardware NVIDIA, status CUDA, dan kapabilitas Tensor Core.
    Mengembalikan dict informasi hardware lengkap.
    """
    info = {
        "cuda_available": False,
        "device_name": "Tidak Terdeteksi",
        "has_tensor_cores": False,
        "tc_generation": None,
        "compute_capability": None,
        "smi_detected": False,
        "smi_name": None,
        "status_code": "cpu_only",
        "message": ""
    }

    if torch.cuda.is_available():
        info["cuda_available"] = True
        info["device_name"] = torch.cuda.get_device_name(0)
        try:
            major, minor = torch.cuda.get_device_capability(0)
            info["compute_capability"] = (major, minor)
            if major >= 9:
                info["has_tensor_cores"] = True
                info["tc_generation"] = "4th/5th Gen Tensor Cores (Hopper/Blackwell)"
            elif major == 8:
                info["has_tensor_cores"] = True
                info["tc_generation"] = "3rd Gen Tensor Cores (Ampere: TF32, FP16, BF16, INT8)"
            elif major == 7 and minor >= 5:
                info["has_tensor_cores"] = True
                info["tc_generation"] = "2nd Gen Tensor Cores (Turing: FP16, INT8)"
            elif major == 7:
                info["has_tensor_cores"] = True
                info["tc_generation"] = "1st Gen Tensor Cores (Volta)"
        except Exception:
            pass

        info["status_code"] = "cuda_ready"
        info["message"] = f"🟢 CUDA & Tensor Core Aktif: {info['device_name']}"
        return info

    # Jika torch.cuda.is_available() False, periksa GPU NVIDIA fisik via nvidia-smi
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2
        )
        if res.returncode == 0 and res.stdout.strip():
            first_line = res.stdout.strip().split("\n")[0].strip()
            info["smi_detected"] = True
            info["smi_name"] = first_line
            info["device_name"] = first_line
            info["status_code"] = "gpu_present_cpu_torch"
            info["message"] = f"⚠️ GPU Terdeteksi: {first_line} (PyTorch build CPU terpasang)"
            return info
    except Exception:
        pass

    info["message"] = "ℹ️ Menggunakan CPU (CUDA GPU tidak terdeteksi)"
    return info


def setup_tensor_core_environment(enable_tc=True, log_callback=print):
    """
    Mengonfigurasi dan mengaktifkan NVIDIA Tensor Core (TF32 matmul, cuDNN TF32, cuDNN benchmark, AMP)
    jika perangkat CUDA GPU tersedia.
    Mengembalikan (device, use_tensor_core_bool).
    """
    hw = get_hardware_acceleration_info()

    if not hw["cuda_available"]:
        if hw["smi_detected"]:
            log_callback(f"[!] GPU Fisik Terdeteksi: {hw['smi_name']}")
            log_callback("    [!] PyTorch yang aktif saat ini adalah build CPU (torch.cuda.is_available() = False).")
            log_callback("    [💡] Untuk mengaktifkan akselerasi Tensor Core penuh di PyTorch:")
            log_callback("         pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126")
            log_callback("    [i] Melanjutkan training dengan fallback CPU (device='cpu').")
        else:
            log_callback("[i] CUDA tidak tersedia, training berjalan pada CPU (device='cpu').")
        return 'cpu', False

    if not enable_tc:
        log_callback(f"[i] Akselerasi Tensor Core dinonaktifkan oleh opsi pengguna (Device: 0/CUDA FP32 standar).")
        return 0, False

    device_name = hw["device_name"]
    cc = hw["compute_capability"]
    cc_str = f"{cc[0]}.{cc[1]}" if cc else "Unknown"

    log_callback("=" * 65)
    log_callback(f"[⚡] MENGAKTIFKAN AKSELERASI NVIDIA TENSOR CORE")
    log_callback(f"    GPU           : {device_name} (Compute Capability: {cc_str})")
    if hw["tc_generation"]:
        log_callback(f"    Arsitektur    : {hw['tc_generation']}")

    # 1. Aktifkan TensorFloat-32 (TF32) pada Tensor Core (Ampere RTX 30xx / Ada RTX 40xx)
    try:
        torch.set_float32_matmul_precision('high')
        log_callback("    [✓] torch.set_float32_matmul_precision('high') -> TF32 Aktif")
    except Exception as e:
        log_callback(f"    [~] Matmul precision note: {e}")

    if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
        torch.backends.cuda.matmul.allow_tf32 = True
        log_callback("    [✓] torch.backends.cuda.matmul.allow_tf32 = True")

    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = True
        log_callback("    [✓] torch.backends.cudnn.allow_tf32 = True")

    # 2. cuDNN Benchmark untuk auto-tuner konvolusi optimal di Tensor Core
    torch.backends.cudnn.benchmark = True
    log_callback("    [✓] torch.backends.cudnn.benchmark = True (Optimalisasi GEMM/Conv)")
    log_callback("    [✓] Automatic Mixed Precision (AMP): FP16 Tensor Cores diaktifkan")
    log_callback("=" * 65)

    return 0, True


def start_training(epochs=80, log_callback=print, model_size='s', use_tensor_core=True):
    """
    Training model YOLO Object Detection untuk Golongan I - VI.
    Mendukung akselerasi NVIDIA Tensor Core (TF32 & FP16 Automatic Mixed Precision).
    """
    root_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(root_dir, "data.yaml")

    if not os.path.exists(yaml_path):
        log_callback(f"[!] File data.yaml tidak ditemukan di: {yaml_path}")
        log_callback("    Gunakan tombol '🔄 Recovery Dataset' untuk rebuild dataset terlebih dahulu.")
        return

    # Cek apakah train/valid ada
    if not _is_dataset_intact(root_dir):
        log_callback("[!] Dataset YOLO (train/valid) tidak lengkap atau kosong!")
        log_callback("    Gunakan tombol '🔄 Recovery Dataset' untuk rebuild dataset dari backup.")
        return

    # 1. Sinkronisasi data custom dari folder Dataset/ dan Dataset_Uji/ jika ada penambahan file baru
    check_and_register_dataset_folder(log_callback=log_callback)
    check_and_register_dataset_uji_folder(log_callback=log_callback)

    # 2. Hitung jumlah dataset deteksi
    train_imgs = os.path.join(root_dir, "train", "images")
    val_imgs = os.path.join(root_dir, "valid", "images")
    test_imgs = os.path.join(root_dir, "test", "images")
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.avif')
    total_train = len([f for f in os.listdir(train_imgs) if f.lower().endswith(exts)]) if os.path.exists(train_imgs) else 0
    total_val = len([f for f in os.listdir(val_imgs) if f.lower().endswith(exts)]) if os.path.exists(val_imgs) else 0
    total_test = len([f for f in os.listdir(test_imgs) if f.lower().endswith(exts)]) if os.path.exists(test_imgs) else 0
    log_callback(f"[+] Dataset siap: {total_train} train, {total_val} val, {total_test} test gambar.")

    # 3. Model YOLOv8 Object Detection (bukan -cls)
    model_pt = f"yolov8{model_size}.pt"

    # 4. Inisialisasi Akselerasi Hardware & NVIDIA Tensor Core
    device, tc_active = setup_tensor_core_environment(enable_tc=use_tensor_core, log_callback=log_callback)

    log_callback(f"\n[+] Memulai training YOLOv8 Object Detection ({model_pt}, {epochs} epoch, imgsz=640)...")
    log_callback(f"    Target Device : {device} | Tensor Core: {'AKTIF (TF32/FP16 AMP)' if tc_active else 'NON-AKTIF'}")

    try:
        runs_dir = os.path.join(root_dir, "runs")
        os.makedirs(runs_dir, exist_ok=True)

        model = YOLO(model_pt)
        train_args = {
            "data": yaml_path,
            "epochs": epochs,
            "imgsz": 640,          # Standar YOLO Object Detection
            "batch": 16,           # Kelipatan 16 optimal untuk memory layout Tensor Core
            "optimizer": 'AdamW',
            "lr0": 0.001,
            "cos_lr": True,
            "warmup_epochs": 3,
            "patience": 20,        # early stopping
            "verbose": False,
            "plots": True,         # simpan grafik dan evaluasi
            "device": device,
            "amp": True if (device != 'cpu' and tc_active) else False, # FP16 Mixed Precision Tensor Core
            "workers": 4 if device != 'cpu' else 2,
            "project": runs_dir,   # Pastikan output run tersimpan di folder runs/ di root direktori
            "name": "train",       # Subfolder di runs/ (train, train2, train3, dst.)
        }
        results = model.train(**train_args)

        best_path = os.path.join(results.save_dir, "weights", "best.pt")
        if os.path.exists(best_path):
            # Salin ke root model_pt (yolov8s.pt) dan juga simpan shortcut di runs/best.pt
            out_pt = os.path.join(root_dir, model_pt)
            shutil.copy(best_path, out_pt)
            shortcut_runs = os.path.join(runs_dir, "best.pt")
            shutil.copy(best_path, shortcut_runs)
            log_callback(f"\n[v] SUCCESS! Model Object Detection tersimpan di:")
            log_callback(f"    - {best_path} (Output Run)")
            log_callback(f"    - {shortcut_runs} (Runs Shortcut)")
            log_callback(f"    - {out_pt} (Root Default Model)")
            log_callback(f"    Siap digunakan langsung untuk real-time tracking.")
        else:
            log_callback("[!] Training selesai tapi weights/best.pt tidak ditemukan.")
    except Exception as e:
        log_callback(f"[!] Error saat training: {e}")

def _golongan_picker(root, on_confirm_cb, default_gol="Golongan I"):
    """Popup dialog modal yang selalu muncul di tengah dan di atas untuk memilih Golongan Kendaraan."""
    golongan_win = tk.Toplevel(root)
    golongan_win.title("Pilih Golongan Kendaraan")
    golongan_win.configure(bg="#0d0f1a")
    golongan_win.resizable(False, False)

    # Pastikan jendela modal dan selalu di depan root
    golongan_win.transient(root)
    golongan_win.attributes("-topmost", True)

    tk.Label(
        golongan_win,
        text="Pilih Golongan Target Kendaraan:",
        font=("Segoe UI", 11, "bold"),
        bg="#0d0f1a",
        fg="#00d4ff"
    ).pack(pady=(15, 8), padx=20)

    selected = tk.StringVar(master=golongan_win, value=default_gol)

    radio_frame = tk.Frame(golongan_win, bg="#12162a", padx=15, pady=8, relief=tk.FLAT)
    radio_frame.pack(padx=20, pady=5, fill=tk.BOTH, expand=True)

    for g in CLASSES:
        rb = tk.Radiobutton(
            radio_frame,
            text=g,
            variable=selected,
            value=g,
            font=("Segoe UI", 10),
            bg="#12162a",
            fg="#f0f4ff",
            selectcolor="#7c3aed",
            activebackground="#12162a",
            activeforeground="#00d4ff",
            cursor="hand2"
        )
        rb.pack(anchor=tk.W, pady=3, padx=10)

    def _confirm():
        val = selected.get()
        golongan_win.destroy()
        on_confirm_cb(val)

    def _on_close():
        golongan_win.destroy()

    golongan_win.protocol("WM_DELETE_WINDOW", _on_close)

    btn_box = tk.Frame(golongan_win, bg="#0d0f1a")
    btn_box.pack(pady=(10, 15))

    tk.Button(
        btn_box,
        text="✓ Konfirmasi",
        bg="#FF9800",
        fg="white",
        font=("Segoe UI", 10, "bold"),
        relief=tk.FLAT,
        padx=18,
        pady=5,
        cursor="hand2",
        command=_confirm
    ).pack(side=tk.LEFT, padx=6)

    tk.Button(
        btn_box,
        text="Batal",
        bg="#2a3050",
        fg="#f0f4ff",
        font=("Segoe UI", 10),
        relief=tk.FLAT,
        padx=12,
        pady=5,
        cursor="hand2",
        command=_on_close
    ).pack(side=tk.LEFT, padx=6)

    # Posisi di tengah root window
    golongan_win.update_idletasks()
    win_w = 340
    win_h = 360
    rx = root.winfo_x() + (root.winfo_width() - win_w) // 2
    ry = root.winfo_y() + (root.winfo_height() - win_h) // 2
    golongan_win.geometry(f"{win_w}x{win_h}+{max(0, rx)}+{max(0, ry)}")

    golongan_win.grab_set()
    golongan_win.focus_force()

def show_train_gui():
    """GUI Training Standar Custom YOLO Model untuk Golongan I - VI dengan Akselerasi NVIDIA Tensor Core"""
    root = tk.Tk()
    root.title("AI Track Kendaraan - Training Model YOLO (NVIDIA Tensor Core Ready)")
    root.geometry("640x700")
    root.configure(bg="#0d0f1a")
    root.eval('tk::PlaceWindow . center')

    tk.Label(
        root,
        text="🏋️  Training Custom Model YOLO (Golongan I - VI)",
        font=("Segoe UI", 13, "bold"),
        bg="#0d0f1a", fg="#00d4ff",
    ).pack(pady=(12, 2))

    # ── Deteksi Hardware Awal ─────────────────────────────────────────
    hw_info = get_hardware_acceleration_info()

    # ── Baris Parameter Epoch & Model ─────────────────────────────────
    param_frame = tk.Frame(root, bg="#0d0f1a")
    param_frame.pack(pady=3)

    tk.Label(param_frame, text="Epochs:", bg="#0d0f1a", fg="#8892b0",
             font=("Segoe UI", 9)).grid(row=0, column=0, padx=6)
    entry_epochs = tk.Entry(param_frame, width=6, font=("Segoe UI", 10),
                            bg="#1a1f35", fg="white", insertbackground="white", relief=tk.FLAT)
    entry_epochs.insert(0, "80")
    entry_epochs.grid(row=0, column=1, padx=4)

    tk.Label(param_frame, text="Model:", bg="#0d0f1a", fg="#8892b0",
             font=("Segoe UI", 9)).grid(row=0, column=2, padx=6)
    model_var = tk.StringVar(value="s")
    for txt, val in [("nano (cepat)", "n"), ("small (rekomendasi)", "s"), ("medium", "m")]:
        tk.Radiobutton(
            param_frame, text=txt, variable=model_var, value=val,
            bg="#0d0f1a", fg="#f0f4ff", selectcolor="#7c3aed",
            activebackground="#0d0f1a", font=("Segoe UI", 9),
        ).grid(row=0, column=3 + ["n", "s", "m"].index(val), padx=3)

    # ── Baris Opsi Akselerasi NVIDIA Tensor Core ──────────────────────
    tc_frame = tk.Frame(root, bg="#12162a", padx=8, pady=4, relief=tk.FLAT)
    tc_frame.pack(fill=tk.X, padx=20, pady=4)

    tc_var = tk.BooleanVar(value=True)
    chk_tc = tk.Checkbutton(
        tc_frame,
        text="⚡ Gunakan NVIDIA Tensor Core (TF32 & FP16 Mixed Precision)",
        variable=tc_var,
        bg="#12162a", fg="#00d4ff", selectcolor="#1a1f35",
        activebackground="#12162a", activeforeground="#00d4ff",
        font=("Segoe UI", 9, "bold")
    )
    chk_tc.pack(side=tk.LEFT)

    def show_tensor_core_info():
        current_hw = get_hardware_acceleration_info()
        info_win = tk.Toplevel(root)
        info_win.title("Informasi Akselerasi Hardware & Tensor Core")
        info_win.geometry("520x360")
        info_win.configure(bg="#0d0f1a")
        info_win.eval('tk::PlaceWindow . center')

        title_lbl = tk.Label(
            info_win, text="⚡ Status NVIDIA Tensor Core",
            font=("Segoe UI", 12, "bold"), bg="#0d0f1a", fg="#00d4ff"
        )
        title_lbl.pack(pady=10)

        detail_text = (
            f"• GPU Fisik: {current_hw['device_name']}\n"
            f"• Status CUDA: {'AKTIF' if current_hw['cuda_available'] else 'TIDAK TERPASANG (PyTorch CPU)'}\n"
            f"• Arsitektur: {current_hw['tc_generation'] or 'Ampere (RTX 3050)'}\n"
            f"• Fitur Tensor Core yang Didukung:\n"
            f"   - TF32 (TensorFloat-32) MatMul: 5x-10x lebih cepat dari FP32\n"
            f"   - cuDNN TF32 Benchmark: Optimasi konvolusi otomatis\n"
            f"   - FP16 Automatic Mixed Precision (AMP): Hemat VRAM 50%\n\n"
        )
        if not current_hw['cuda_available'] and current_hw['smi_detected']:
            detail_text += (
                "💡 CATATAN INSTALASI:\n"
                "GPU NVIDIA RTX 3050 Anda terdeteksi dengan baik!\n"
                "Untuk mengaktifkan CUDA pada PyTorch 3.14 di Windows, jalankan:\n\n"
                "pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126\n\n"
                "Setelah itu, training YOLO akan otomatis langsung menggunakan Tensor Core."
            )
        elif current_hw['cuda_available']:
            detail_text += "✅ Sistem 100% siap melatih dengan NVIDIA Tensor Core!"
        else:
            detail_text += "ℹ️ Sistem saat ini menggunakan akselerasi CPU."

        msg_box = tk.Text(
            info_win, height=12, width=60, font=("Consolas", 9),
            bg="#161b30", fg="#f0f4ff", padx=10, pady=10, relief=tk.FLAT
        )
        msg_box.insert(tk.END, detail_text)
        msg_box.config(state=tk.DISABLED)
        msg_box.pack(pady=5, padx=15)

        tk.Button(
            info_win, text="Tutup", bg="#7c3aed", fg="white",
            font=("Segoe UI", 9, "bold"), relief=tk.FLAT, padx=15, pady=4,
            command=info_win.destroy
        ).pack(pady=10)

    btn_tc_info = tk.Button(
        tc_frame, text="ℹ️ Status GPU",
        font=("Segoe UI", 8), bg="#2a3050", fg="#f0f4ff",
        relief=tk.FLAT, padx=6, pady=2, cursor="hand2", command=show_tensor_core_info
    )
    btn_tc_info.pack(side=tk.RIGHT)

    # ── Baris Pilihan Golongan Target ─────────────────────────────────
    gol_frame = tk.Frame(root, bg="#12162a", padx=10, pady=5, relief=tk.FLAT)
    gol_frame.pack(fill=tk.X, padx=20, pady=4)

    tk.Label(
        gol_frame,
        text="🎯 Target Golongan (untuk Tambah Foto / Video):",
        bg="#12162a", fg="#00d4ff",
        font=("Segoe UI", 9, "bold")
    ).pack(side=tk.LEFT, padx=(0, 10))

    gol_var = tk.StringVar(value="Golongan I")
    gol_combo = ttk.Combobox(
        gol_frame,
        textvariable=gol_var,
        values=CLASSES,
        state="readonly",
        width=16,
        font=("Segoe UI", 9, "bold")
    )
    gol_combo.pack(side=tk.LEFT)

    # Status Bar Hardware kecil
    hw_status_lbl = tk.Label(
        root,
        text=hw_info["message"],
        font=("Segoe UI", 8),
        bg="#0d0f1a",
        fg="#00d4ff" if hw_info["cuda_available"] else ("#f59e0b" if hw_info["smi_detected"] else "#8892b0"),
        anchor=tk.W, padx=22
    )
    hw_status_lbl.pack(fill=tk.X, pady=(0, 2))

    log_text = tk.Text(
        root, height=14, width=74, font=("Consolas", 9),
        bg="#12162a", fg="#f0f4ff", insertbackground="white", relief=tk.FLAT,
    )
    log_text.pack(pady=6, padx=20)

    def log(msg):
        log_text.insert(tk.END, msg + "\n")
        log_text.see(tk.END)
        root.update_idletasks()

    def get_epochs():
        try:
            return int(entry_epochs.get())
        except ValueError:
            return 80

    all_btns = []

    def set_buttons(state):
        for b in all_btns:
            try:
                b.config(state=state)
            except Exception:
                pass

    def on_start():
        set_buttons(tk.DISABLED)
        log_text.delete("1.0", tk.END)
        def worker():
            start_training(
                epochs=get_epochs(),
                log_callback=log,
                model_size=model_var.get(),
                use_tensor_core=tc_var.get()
            )
            set_buttons(tk.NORMAL)
        threading.Thread(target=worker, daemon=True).start()

    def on_foto_train():
        image_paths = filedialog.askopenfilenames(
            title="Pilih Foto Sampel Kendaraan",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not image_paths:
            return
        def confirm(golongan):
            gol_var.set(golongan)
            set_buttons(tk.DISABLED)
            log_text.delete("1.0", tk.END)
            def worker():
                add_images_to_dataset(list(image_paths), golongan, log_callback=log)
                start_training(
                    epochs=get_epochs(),
                    log_callback=log,
                    model_size=model_var.get(),
                    use_tensor_core=tc_var.get()
                )
                set_buttons(tk.NORMAL)
            threading.Thread(target=worker, daemon=True).start()
        _golongan_picker(root, confirm, default_gol=gol_var.get())

    def on_video_train():
        video_path = filedialog.askopenfilename(
            title="Pilih Video Sampel Kendaraan",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if not video_path:
            return
        def confirm(golongan):
            gol_var.set(golongan)
            set_buttons(tk.DISABLED)
            log_text.delete("1.0", tk.END)
            def worker():
                extract_frames_from_video(video_path, golongan, fps_interval=30, log_callback=log)
                start_training(
                    epochs=get_epochs(),
                    log_callback=log,
                    model_size=model_var.get(),
                    use_tensor_core=tc_var.get()
                )
                set_buttons(tk.NORMAL)
            threading.Thread(target=worker, daemon=True).start()
        _golongan_picker(root, confirm, default_gol=gol_var.get())

    def on_recovery():
        """Rebuild dataset dari Dataset_Original/ + Dataset/ + Dataset_Uji/."""
        confirm = messagebox.askyesno(
            "Recovery Dataset",
            "Ini akan me-rebuild ulang folder train/, valid/, dan test/ "
            "dari Dataset_Original/ (Roboflow) + Dataset/ (custom) + Dataset_Uji/ (uji).\n\n"
            "Folder train/, valid/, dan test/ yang ada sekarang akan di-timpa.\n\n"
            "Lanjutkan?"
        )
        if not confirm:
            return
        set_buttons(tk.DISABLED)
        log_text.delete("1.0", tk.END)
        def worker():
            root_dir = os.path.dirname(os.path.abspath(__file__))
            # Bersihkan train/, valid/, dan test/ lama
            for folder in ["train", "valid", "test"]:
                folder_path = os.path.join(root_dir, folder)
                if os.path.exists(folder_path):
                    shutil.rmtree(folder_path)
                    log(f"[~] Folder {folder}/ dihapus.")
            # Rebuild dari sumber
            rebuilt = rebuild_dataset_from_sources(log_callback=log)
            if rebuilt > 0:
                log("\n[✓] Recovery selesai! Dataset siap untuk training.")
                log("    Klik '🚀 Mulai Training YOLO' untuk melatih model.")
            else:
                log("\n[!] Recovery gagal — tidak ada data sumber.")
                log("    Pastikan Dataset/ atau Dataset_Original/ berisi data.")
            set_buttons(tk.NORMAL)
        threading.Thread(target=worker, daemon=True).start()

    btn_frame = tk.Frame(root, bg="#0d0f1a")
    btn_frame.pack(pady=4)

    btn_train = tk.Button(
        btn_frame, text="🚀 Mulai Training YOLO",
        font=("Segoe UI", 9, "bold"), bg="#FF9800", fg="white",
        relief=tk.FLAT, padx=12, pady=6, cursor="hand2", command=on_start,
    )
    btn_train.grid(row=0, column=0, padx=6)
    all_btns.append(btn_train)

    btn_foto = tk.Button(
        btn_frame, text="🖼️ Tambah Foto & Train",
        font=("Segoe UI", 9, "bold"), bg="#9C27B0", fg="white",
        relief=tk.FLAT, padx=12, pady=6, cursor="hand2", command=on_foto_train,
    )
    btn_foto.grid(row=0, column=1, padx=6)
    all_btns.append(btn_foto)

    btn_video = tk.Button(
        btn_frame, text="🎥 Ekstrak Video & Train",
        font=("Segoe UI", 9, "bold"), bg="#E91E63", fg="white",
        relief=tk.FLAT, padx=12, pady=6, cursor="hand2", command=on_video_train,
    )
    btn_video.grid(row=0, column=2, padx=6)
    all_btns.append(btn_video)

    # ── Baris tombol kedua: Recovery ──────────────────────────────────
    btn_frame2 = tk.Frame(root, bg="#0d0f1a")
    btn_frame2.pack(pady=2)

    btn_recovery = tk.Button(
        btn_frame2, text="🔄 Recovery Dataset",
        font=("Segoe UI", 9, "bold"), bg="#00897B", fg="white",
        relief=tk.FLAT, padx=12, pady=5, cursor="hand2", command=on_recovery,
    )
    btn_recovery.pack()
    all_btns.append(btn_recovery)

    status_lbl = tk.Label(
        root, text="Siap. Klik 'Mulai Training YOLO' untuk melatih model deteksi.",
        font=("Segoe UI", 9), bg="#0a0d1a", fg="#8892b0",
        anchor=tk.W, padx=20, pady=5,
    )
    status_lbl.pack(fill=tk.X, side=tk.BOTTOM)

    root.mainloop()


if __name__ == "__main__":
    show_train_gui()
