import os
import cv2
import numpy as np
import train
from ultralytics import YOLO

# Sinkronkan path data.yaml ke lokasi folder saat ini agar portabel di path mana saja
def sync_data_yaml_path():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(root_dir, "data.yaml")
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            root_unix = root_dir.replace("\\", "/")
            if lines and lines[0].startswith("path:"):
                current_p = lines[0].split("path:", 1)[1].strip()
                if current_p != root_unix:
                    lines[0] = f"path: {root_unix}\n"
                    with open(yaml_path, "w", encoding="utf-8") as f:
                        f.writelines(lines)
        except Exception:
            pass

sync_data_yaml_path()

# Direktori output hasil deteksi
HASIL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Hasil")

# Daftar Golongan Kendaraan di Indonesia (Golongan I - VI)
CLASSES = [
    "Golongan I",   # Sedan, Jip, Pick Up, Bus, Mobil Pribadi
    "Golongan II",  # Truk Besar 2 Gandar
    "Golongan III", # Truk Besar 3 Gandar
    "Golongan IV",  # Truk Besar 4 Gandar
    "Golongan V",   # Truk Besar 5 Gandar atau Lebih
    "Golongan VI"   # Sepeda Motor / Roda 2
]

# Warna Bounding Box Khusus Tiap Golongan (BGR Format)
GOLONGAN_COLORS = {
    "Golongan I":   (0, 255, 0),     # Hijau
    "Golongan II":  (255, 128, 0),   # Oranye
    "Golongan III": (255, 255, 0),   # Cyan
    "Golongan IV":  (0, 255, 255),   # Kuning
    "Golongan V":   (0, 69, 255),    # Merah-Oranye
    "Golongan VI":  (255, 0, 255),   # Magenta
}

# Fallback Pemetaan COCO Pretrained jika belum ada Custom Trained Model
COCO_CLASS_MAPPING = {
    "car": "Golongan I",
    "bus": "Golongan I",
    "truck": "Golongan II",
    "motorcycle": "Golongan VI",
    "bicycle": "Golongan VI"
}

ROMAN_GOLONGAN_MAP = {
    "GOLONGAN I": "Golongan I", "GOL I": "Golongan I", "GOL 1": "Golongan I", "GOLONGAN 1": "Golongan I",
    "GOLONGAN II": "Golongan II", "GOL II": "Golongan II", "GOL 2": "Golongan II", "GOLONGAN 2": "Golongan II",
    "GOLONGAN III": "Golongan III", "GOL III": "Golongan III", "GOL 3": "Golongan III", "GOLONGAN 3": "Golongan III",
    "GOLONGAN IV": "Golongan IV", "GOL IV": "Golongan IV", "GOL 4": "Golongan IV", "GOLONGAN 4": "Golongan IV",
    "GOLONGAN V": "Golongan V", "GOL V": "Golongan V", "GOL 5": "Golongan V", "GOLONGAN 5": "Golongan V",
    "GOLONGAN VI": "Golongan VI", "GOL VI": "Golongan VI", "GOL 6": "Golongan VI", "GOLONGAN 6": "Golongan VI",
}

def normalize_golongan_name(raw_name, cls_id=None):
    """Normalisasi nama kelas Roboflow / YOLO ke standar Golongan I - VI tanpa substring bug."""
    if not raw_name and cls_id is None:
        return None
    r = str(raw_name).strip().upper().replace("_", " ")
    r_single = " ".join(r.split())
    if r_single in ROMAN_GOLONGAN_MAP:
        return ROMAN_GOLONGAN_MAP[r_single]
    r_nospace = r.replace(" ", "")
    for k, v in ROMAN_GOLONGAN_MAP.items():
        if r_nospace == k.replace(" ", ""):
            return v
    if cls_id is not None and 0 <= cls_id < len(CLASSES):
        return CLASSES[cls_id]
    return None

DEFAULT_YOLO_MODELS = ["yolov8s.pt", "yolov8m.pt", "yolov8n.pt"]

_COCO_FALLBACK_MODEL = None

def get_coco_fallback_model():
    """
    Load dan cache model COCO pretrained (yolov8s/yolov8n) dari folder coco_model/
    khusus untuk fallback Golongan VI (sepeda motor / roda 2)
    serta membantu deteksi banyak kendaraan (multi-vehicle) secara presisi seperti di v4.
    """
    global _COCO_FALLBACK_MODEL
    if _COCO_FALLBACK_MODEL is not None:
        return _COCO_FALLBACK_MODEL

    root_dir = os.path.dirname(os.path.abspath(__file__))
    coco_dir = os.path.join(root_dir, "coco_model")
    cand_coco = [
        os.path.join(coco_dir, "yolov8s.pt"),
        os.path.join(coco_dir, "yolov8n.pt"),
        os.path.join(coco_dir, "best.pt"),
        os.path.join(root_dir, "yolov8s.pt"),
        os.path.join(root_dir, "yolov8n.pt"),
        "yolov8s.pt",
        "yolov8n.pt"
    ]
    for cp in cand_coco:
        if os.path.exists(cp):
            try:
                m = YOLO(cp)
                if 'motorcycle' in m.names.values():
                    print(f"[+] Loaded COCO Fallback Model (Golongan VI): {cp}")
                    _COCO_FALLBACK_MODEL = m
                    return m
            except Exception:
                pass

    try:
        m = YOLO("yolov8s.pt")
        if 'motorcycle' in m.names.values():
            _COCO_FALLBACK_MODEL = m
            return m
    except Exception:
        try:
            m = YOLO("yolov8n.pt")
            _COCO_FALLBACK_MODEL = m
            return m
        except Exception:
            return None

def get_classifier_model():
    """Deteksi dan klasifikasi Golongan I - VI terintegrasi langsung pada model YOLOv8."""
    return None, None

def get_detector_model(model_mode="auto"):
    """
    Load model detektor YOLOv8:
    model_mode:
      - 'auto'   : Cek runs/best.pt, jika tidak ada fallback ke model standar COCO.
      - 'custom' : Menggunakan runs/best.pt hasil training lokal (dilengkapi COCO fallback).
      - 'base'   : Menggunakan yolov8s.pt / yolov8n.pt pretrained COCO (mendeteksi multi-kendaraan dengan presisi tinggi).
    """
    root_dir = os.path.dirname(os.path.abspath(__file__))
    coco_dir = os.path.join(root_dir, "coco_model")

    # 1. JIKA USER MEMILIH MODEL BASE / STANDAR SECARA EKSPLISIT
    if model_mode in ["base", "coco", "default"]:
        cand_base = [
            os.path.join(coco_dir, "yolov8s.pt"),
            os.path.join(coco_dir, "yolov8n.pt"),
            os.path.join(coco_dir, "best.pt"),
            os.path.join(root_dir, "yolov8s.pt"),
            os.path.join(root_dir, "yolov8n.pt"),
            "yolov8s.pt",
            "yolov8n.pt"
        ]
        for cp in cand_base:
            if os.path.exists(cp):
                try:
                    m = YOLO(cp)
                    if 'motorcycle' in m.names.values():
                        print(f"[+] Loaded Model Dasar Pretrained COCO (Multi-Vehicle): {cp}")
                        return m
                except Exception:
                    pass
        try:
            return YOLO("yolov8s.pt")
        except Exception:
            return YOLO("yolov8n.pt")

    # 2. PRIORITAS CUSTOM (runs/best.pt)
    if model_mode in ["auto", "custom"]:
        runs_dir = os.path.join(root_dir, "runs")
        if os.path.exists(runs_dir):
            best_files = []
            for r, _, files in os.walk(runs_dir):
                for f in files:
                    if f == "best.pt":
                        p = os.path.join(r, f)
                        best_files.append((os.path.getmtime(p), p))
            if best_files:
                best_files.sort(reverse=True)
                for _, chosen in best_files:
                    try:
                        m = YOLO(chosen)
                        print(f"[+] Loaded Model Hasil Training dari runs/: {chosen}")
                        return m
                    except Exception as e:
                        print(f"[!] Gagal me-load model dari runs/ ({chosen}): {e}")

        # Cek file best.pt langsung di root folder
        root_best = os.path.join(root_dir, "best.pt")
        if os.path.exists(root_best):
            try:
                m = YOLO(root_best)
                print(f"[+] Loaded Model Custom dari root: {root_best}")
                return m
            except Exception as e:
                print(f"[!] Gagal me-load best.pt dari root: {e}")

    # Fallback ke model dasar pretrained COCO
    cand_fb = [
        os.path.join(coco_dir, "yolov8s.pt"),
        os.path.join(coco_dir, "yolov8n.pt"),
        os.path.join(coco_dir, "best.pt"),
        os.path.join(root_dir, "yolov8s.pt"),
        "yolov8s.pt"
    ]
    for cp in cand_fb:
        if os.path.exists(cp):
            try:
                m = YOLO(cp)
                if 'motorcycle' in m.names.values():
                    print(f"[+] Loaded Detector Model (COCO Fallback): {cp}")
                    return m
            except Exception:
                pass

    return YOLO("yolov8s.pt")

def calculate_box_overlap(boxA, boxB):
    """
    Hitung Intersection over Union (IoU) dan containment antara dua bounding box.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    if inter_area <= 0:
        return 0.0, 0.0

    areaA = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    iou = inter_area / float(areaA + areaB - inter_area)
    iomin = inter_area / float(min(areaA, areaB))

    return iou, iomin

def filter_colliding_boxes(detections, overlap_threshold=0.60):
    """
    Saring HANYA duplikasi prediksi dari KENDARAAN YANG SAMA (IoU tinggi).
    Mempertahankan kendaraan berbeda yang saling berdekatan, beriringan, atau di lajur berbeda
    sehingga sistem dapat mendeteksi banyak kendaraan sekaligus (multi-vehicle detection).
    Prioritaskan deteksi Golongan III - V agar tidak tertindih oleh deteksi model COCO default.
    """
    if not detections:
        return []

    # Beri prioritas tertinggi untuk deteksi Custom Model Golongan III - V agar tidak tertelan box COCO
    def sort_key(d):
        is_custom_heavy = 1 if (d.get("source") == "custom" and d.get("golongan_name") in ("Golongan III", "Golongan IV", "Golongan V")) else 0
        return (is_custom_heavy, d.get("conf", 0.0))

    sorted_dets = sorted(detections, key=sort_key, reverse=True)
    kept_dets = []

    for det in sorted_dets:
        box = det["box"]
        is_duplicate = False
        for kept in kept_dets:
            iou, iomin = calculate_box_overlap(box, kept["box"])
            # Jika objek yang sudah disimpan adalah Golongan III - V dari custom model,
            # jangan biarkan box COCO default menindih
            if kept.get("golongan_name") in ("Golongan III", "Golongan IV", "Golongan V") and det.get("source") == "coco_fallback":
                if iou >= 0.15 or iomin >= 0.20:
                    is_duplicate = True
                    break

            # Hanya eliminasi jika dua box adalah duplikat dari kendaraan yang sama (IoU tinggi)
            # ATAU jika satu box kecil hampir tertelan seluruhnya di dalam box besar dengan kelas yang sama
            if iou >= overlap_threshold or (det["golongan_name"] == kept["golongan_name"] and iomin >= 0.88):
                is_duplicate = True
                break

        if not is_duplicate:
            kept_dets.append(det)

    return kept_dets

def draw_vehicle_boxes(detector, classifier, frame, conf=0.15, overlap_threshold=0.60, return_stats=False):
    """
    Deteksi bounding box semua kendaraan di frame,
    lalu klasifikasikan ke Golongan I - VI secara akurat.
    Mendukung deteksi banyak kendaraan secara bersamaan (multi-vehicle detection)
    seperti di v4, dengan fallback Golongan VI ke model COCO pretrained untuk
    menghasilkan bounding box yang presisi, rapi, dan tidak salah mendeteksi.
    """
    stats = {g: 0 for g in CLASSES}
    h_frame, w_frame = frame.shape[:2]
    frame_area = max(1, h_frame * w_frame)

    # Cek apakah detector memiliki kelas khusus Golongan I - VI (Model Custom)
    # Mendukung format "Golongan I" - "Golongan VI" maupun "GOL I" - "GOL VI" (Roboflow)
    has_tol_classes = any(
        ("golongan" in str(v).lower() or str(v).lower().startswith("gol"))
        for v in getattr(detector, 'names', {}).values()
    )
    is_custom_model = has_tol_classes

    # Deteksi kendaraan menggunakan detector YOLOv8 (conf 0.15 agar dapat scan banyak kendaraan seperti di v4)
    results = detector.predict(frame, conf=conf, verbose=False)

    candidates = []

    if results and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            raw_name = str(detector.names.get(cls_id, "")).strip()
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Batasi koordinat ke dalam batas frame
            x1 = max(0, min(x1, w_frame - 1))
            y1 = max(0, min(y1, h_frame - 1))
            x2 = max(0, min(x2, w_frame))
            y2 = max(0, min(y2, h_frame))

            box_w = x2 - x1
            box_h = y2 - y1
            box_area = box_w * box_h

            # Cegah anomali box raksasa memenuhi seluruh layar (> 85% area frame)
            if box_area > 0.85 * frame_area:
                continue

            # Abaikan kotak yang terlalu kecil / noise kamera (< 10px)
            if box_w < 10 or box_h < 10:
                continue

            conf_val = float(box.conf[0])
            pred_conf = conf_val
            raw_coco_name = raw_name.lower()

            # Normalisasi nama golongan
            golongan_name = normalize_golongan_name(raw_name, cls_id if is_custom_model else None)

            # Jika model custom classifier tersedia, klasifikasikan ROI
            if golongan_name is None and classifier is not None and box_h > 10 and box_w > 10:
                roi = frame[y1:y2, x1:x2]
                if roi.shape[0] > 10 and roi.shape[1] > 10:
                    try:
                        cls_res = classifier.predict(roi, verbose=False)
                        if cls_res and hasattr(cls_res[0], 'probs') and cls_res[0].probs is not None:
                            top_idx = int(cls_res[0].probs.top1)
                            top_conf = float(cls_res[0].probs.top1conf)

                            # Ambil nama golongan dari classifier
                            if hasattr(classifier, 'names') and top_idx in classifier.names:
                                golongan_name = str(classifier.names[top_idx])
                            elif top_idx < len(CLASSES):
                                golongan_name = CLASSES[top_idx]

                            pred_conf = top_conf
                    except Exception:
                        pass

            # Penanganan Klasifikasi / Fallback berdasarkan tipe model seperti di v4
            if is_custom_model:
                # MODEL CUSTOM: Fallback HANYA untuk Golongan I (car/bus) dan Golongan VI (motorcycle/bicycle)
                # Golongan II - V (Truk) murni dari model custom
                if golongan_name is None:
                    coco_gol = COCO_CLASS_MAPPING.get(raw_coco_name)
                    if coco_gol in ("Golongan I", "Golongan VI"):
                        golongan_name = coco_gol

                if not golongan_name or golongan_name not in CLASSES:
                    continue
            else:
                # MODEL DEFAULT COCO: Jalankan pemetaan COCO fallback lengkap
                if golongan_name is None:
                    coco_gol = COCO_CLASS_MAPPING.get(raw_coco_name)
                    if coco_gol in CLASSES:
                        golongan_name = coco_gol
                    elif raw_coco_name in ["vehicle", "automobile"]:
                        golongan_name = "Golongan I"
                    else:
                        continue

            candidates.append({
                "box": (x1, y1, x2, y2),
                "golongan_name": golongan_name,
                "conf": pred_conf,
                "source": "custom" if is_custom_model else "coco"
            })

    # ── FALLBACK KE COCO KHUSUS GOLONGAN VI & SCAN BANYAK KENDARAAN ──────
    # Jika menggunakan Model Custom, aktifkan fallback ke COCO pretrained:
    # 1. Golongan VI (sepeda motor / roda 2) digantikan atau ditambahkan dari COCO
    #    agar bounding box rapi & presisi seperti di v4.
    # 2. Kendaraan lain yang terlewat oleh custom model akan dilengkapi oleh COCO
    #    sehingga mampu men-scan banyak kendaraan di berbagai situasi lalu lintas.
    if is_custom_model:
        coco_fb = get_coco_fallback_model()
        if coco_fb is not None:
            coco_res = coco_fb.predict(frame, conf=conf, verbose=False)
            if coco_res and len(coco_res[0].boxes) > 0:
                for box in coco_res[0].boxes:
                    c_id = int(box.cls[0])
                    c_raw = str(coco_fb.names.get(c_id, "")).lower()
                    if c_raw not in COCO_CLASS_MAPPING:
                        continue
                    c_gol = COCO_CLASS_MAPPING[c_raw]
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                    bx1 = max(0, min(bx1, w_frame - 1))
                    by1 = max(0, min(by1, h_frame - 1))
                    bx2 = max(0, min(bx2, w_frame))
                    by2 = max(0, min(by2, h_frame))
                    b_conf = float(box.conf[0])
                    b_box = (bx1, by1, bx2, by2)

                    if c_gol == "Golongan VI":
                        # Cegah COCO Golongan VI menindih area truk Golongan III - V
                        overlap_g3_5 = False
                        for cand in candidates:
                            if cand.get("golongan_name") in ("Golongan III", "Golongan IV", "Golongan V"):
                                iou, iomin = calculate_box_overlap(b_box, cand["box"])
                                if iou >= 0.20 or iomin >= 0.35:
                                    overlap_g3_5 = True
                                    break
                        if overlap_g3_5:
                            continue

                        # Khusus Golongan VI: Gantikan deteksi custom yang tumpang tindih
                        # dengan bounding box COCO yang lebih presisi, atau tambahkan jika belum ada.
                        replaced = False
                        for idx, cand in enumerate(candidates):
                            iou, _ = calculate_box_overlap(b_box, cand["box"])
                            if iou >= 0.35 and cand["golongan_name"] == "Golongan VI":
                                candidates[idx] = {
                                    "box": b_box,
                                    "golongan_name": c_gol,
                                    "conf": b_conf,
                                    "source": "coco_fallback"
                                }
                                replaced = True
                                break
                        if not replaced:
                            candidates.append({
                                "box": b_box,
                                "golongan_name": c_gol,
                                "conf": b_conf,
                                "source": "coco_fallback"
                            })
                    else:
                        # Kendaraan lain (Golongan I / II): tambahkan jika belum terdeteksi oleh custom
                        # JANGAN pernah tindih jika area tersebut sudah terdeteksi Golongan III, IV, atau V
                        overlap = False
                        for cand in candidates:
                            iou, iomin = calculate_box_overlap(b_box, cand["box"])
                            # Jika area sudah terdeteksi Golongan III - V, tolak box COCO default
                            if cand.get("golongan_name") in ("Golongan III", "Golongan IV", "Golongan V"):
                                if iou >= 0.15 or iomin >= 0.20:
                                    overlap = True
                                    break
                            elif iou >= 0.40:
                                overlap = True
                                break
                        if not overlap:
                            candidates.append({
                                "box": b_box,
                                "golongan_name": c_gol,
                                "conf": b_conf,
                                "source": "coco_fallback"
                            })

    # Saring HANYA box duplikasi nyata (overlap_threshold=0.60)
    # Ini memastikan semua kendaraan berbeda di jalan tetap terdeteksi (Multi-Vehicle)
    filtered_detections = filter_colliding_boxes(candidates, overlap_threshold=overlap_threshold)

    for det in filtered_detections:
        x1, y1, x2, y2 = det["box"]
        golongan_name = det["golongan_name"]
        pred_conf = det["conf"]

        # Increment statistik
        stats[golongan_name] = stats.get(golongan_name, 0) + 1

        # Warna & Label Box
        color = GOLONGAN_COLORS.get(golongan_name, (0, 255, 0))

        # Gambar Bounding Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label Tag
        label_text = f"{golongan_name} ({pred_conf*100:.0f}%)"
        (w_txt, h_txt), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)

        # Background Box Tag
        tag_y1 = max(y1 - 22, 0)
        tag_y2 = tag_y1 + 22
        cv2.rectangle(frame, (x1, tag_y1), (x1 + w_txt + 8, tag_y2), color, -1)

        # Teks Tag (Warna Hitam / Putih kontras)
        text_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
        cv2.putText(frame, label_text, (x1 + 4, tag_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2)

    draw_header_overlay(frame, stats, is_custom_model=is_custom_model)

    if return_stats:
        return frame, stats
    return frame

def draw_header_overlay(frame, stats, is_custom_model=False):
    """Draw semi-transparent header bar showing real-time Golongan count statistics."""
    h, w = frame.shape[:2]
    header_h = 55

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    total_count = sum(stats.values())
    status_str = "Model Custom" if is_custom_model else "Model YOLOv8 Default (COCO Mode)"

    # Header Bar Row 1: Total & Status Mode
    title_txt = f"TOTAL DETEKSI: {total_count} Kendaraan | Mode: {status_str}"
    cv2.putText(frame, title_txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # Header Bar Row 2: Rincian Golongan I - VI
    summary_parts = []
    for g in CLASSES:
        cnt = stats.get(g, 0)
        summary_parts.append(f"{g.replace('Golongan ', 'Gol ')}: {cnt}")

    summary_txt = " | ".join(summary_parts)
    cv2.putText(frame, summary_txt, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def process_video(source, save_output=False, model_mode="auto"):
    """Proses deteksi & tracking kendaraan pada Video / Kamera live."""
    detector = get_detector_model(model_mode=model_mode)
    classifier, model_path = get_classifier_model()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[!] Gagal membuka sumber video / kamera.")
        return

    writer = None
    if save_output:
        os.makedirs(HASIL_DIR, exist_ok=True)
        filename = f"hasil_{os.path.basename(str(source)) if isinstance(source, str) else 'kamera.mp4'}"
        out_path = os.path.join(HASIL_DIR, filename)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        print(f"[+] Output video disimpan ke: {out_path}")

    window_name = "AI Track Kendaraan (Golongan I - VI)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("[+] Tekan 'q' pada keyboard ATAU tombol (X) untuk keluar.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = draw_vehicle_boxes(detector, classifier, frame, conf=0.15)

        if writer:
            writer.write(frame)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("[*] Keluar dari aplikasi.")
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

def process_image(image_paths, save_output=True, model_mode="auto"):
    """Proses deteksi & tracking kendaraan pada satu atau beberapa Foto."""
    detector = get_detector_model(model_mode=model_mode)
    classifier, model_path = get_classifier_model()

    if isinstance(image_paths, str):
        image_paths = [image_paths]

    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[!] Gagal membaca foto: {img_path}")
            continue

        frame = draw_vehicle_boxes(detector, classifier, frame, conf=0.15)

        if save_output:
            os.makedirs(HASIL_DIR, exist_ok=True)
            filename = f"hasil_{os.path.basename(img_path)}"
            out_path = os.path.join(HASIL_DIR, filename)
            cv2.imwrite(out_path, frame)
            print(f"[+] Output foto disimpan ke: {out_path}")

        window_name = f"Hasil Deteksi: {os.path.basename(img_path)}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, frame)
        print("[+] Tekan sembarang tombol keyboard untuk foto berikutnya / keluar.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def show_gui():
    """GUI Utama Aplikasi AI Track Kendaraan (Golongan I - VI)"""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.title("AI Track Kendaraan (Golongan I - VI)")
    root.geometry("700x260")
    root.eval('tk::PlaceWindow . center')

    tk.Label(root, text="🚗 AI Track Kendaraan Indonesia (Golongan I - VI)", font=("Arial", 12, "bold")).pack(pady=10)

    # ── Pilihan Model ─────────────────────────────────────────────
    mode_frame = tk.LabelFrame(root, text=" Pilihan Model AI ", font=("Arial", 9, "bold"), fg="#1976D2")
    mode_frame.pack(pady=4, padx=20, fill=tk.X)

    model_var = tk.StringVar(value="base")
    r1 = tk.Radiobutton(
        mode_frame,
        text="Model Standar YOLO (coco_model/yolov8s.pt - Default COCO)",
        variable=model_var, value="base", font=("Arial", 9, "bold"), fg="#2E7D32"
    )
    r1.pack(anchor=tk.W, padx=12, pady=2)

    r2 = tk.Radiobutton(
        mode_frame,
        text="Model Custom Tol (runs/best.pt - Hasil Training Mandiri dan COCO Fallback Golongan VI)",
        variable=model_var, value="custom", font=("Arial", 9)
    )
    r2.pack(anchor=tk.W, padx=12, pady=2)

    tk.Label(root, text="Pilih Mode Input untuk Deteksi & Tracking Kendaraan:", font=("Arial", 9)).pack(pady=4)

    def on_camera():
        m_mode = model_var.get()
        root.destroy()
        process_video(0, save_output=False, model_mode=m_mode)

    def on_upload_image():
        file_paths = filedialog.askopenfilenames(
            title="Pilih Foto Kendaraan",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if file_paths:
            m_mode = model_var.get()
            root.destroy()
            process_image(list(file_paths), save_output=True, model_mode=m_mode)

    def on_upload():
        file_path = filedialog.askopenfilename(
            title="Pilih File Video",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if file_path:
            m_mode = model_var.get()
            root.destroy()
            process_video(file_path, save_output=True, model_mode=m_mode)

    def on_train():
        root.destroy()
        train.show_train_gui()

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=8)

    tk.Button(btn_frame, text="Kamera Live",  font=("Arial", 10, "bold"), width=15, command=on_camera,       bg="#4CAF50", fg="white").grid(row=0, column=0, padx=5)
    tk.Button(btn_frame, text="Upload Foto",   font=("Arial", 10, "bold"), width=15, command=on_upload_image, bg="#9C27B0", fg="white").grid(row=0, column=1, padx=5)
    tk.Button(btn_frame, text="Upload Video",  font=("Arial", 10, "bold"), width=15, command=on_upload,       bg="#2196F3", fg="white").grid(row=0, column=2, padx=5)
    tk.Button(btn_frame, text="Custom Train",  font=("Arial", 10, "bold"), width=15, command=on_train,        bg="#FF9800", fg="white").grid(row=0, column=3, padx=5)

    root.mainloop()

if __name__ == "__main__":
    show_gui()
