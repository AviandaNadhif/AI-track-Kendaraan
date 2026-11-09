# -*- coding: utf-8 -*-
"""
evaluate_golongan.py
====================
Pengujian 10 citra per golongan (Golongan I - VI) untuk naskah jurnal.
Letakkan file ini di folder yang sama dengan Track.py, lalu siapkan:

    Dataset_Uji/
        Golongan I/    (10 citra)
        Golongan II/   (10 citra)
        ...
        Golongan VI/   (10 citra)

Jalankan:
    python evaluate_golongan.py                 # pengujian citra + kecepatan
    python evaluate_golongan.py --val           # + metrik validasi Ultralytics (Tabel validasi)
    python evaluate_golongan.py --device cpu    # ulangi untuk konfigurasi CPU
    python evaluate_golongan.py --device 0      # GPU

Keluaran (folder Hasil_Uji/): log_pengujian.csv, hasil_uji.json, ringkasan.txt
Kirim hasil_uji.json + log_pengujian.csv agar tabel naskah bisa diisi otomatis.
"""
import os, csv, json, math, time, argparse
import cv2
import numpy as np

from PIL import Image

from Track import (get_detector_model, normalize_golongan_name,
                   filter_colliding_boxes, CLASSES)

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif")
CONF = 0.15
OVERLAP = 0.30


def load_image(path):
    """Membaca citra dengan OpenCV, dengan fallback PIL jika format khusus (misal .avif)."""
    img = cv2.imread(path)
    if img is None:
        try:
            with Image.open(path) as pil_img:
                img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    return img


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - m), min(1.0, c + m))


def predict_image(detector, img, device):
    """Mengembalikan (golongan_prediksi | 'TD', keyakinan | None)."""
    res = detector.predict(img, conf=CONF, agnostic_nms=True, verbose=False, device=device)
    cands = []
    if res and len(res[0].boxes) > 0:
        has_tol = any("golongan" in str(v).lower() or str(v).lower().startswith("gol")
                      for v in detector.names.values())
        for b in res[0].boxes:
            cid = int(b.cls[0])
            name = normalize_golongan_name(str(detector.names.get(cid, "")).strip(),
                                           cid if has_tol else None)
            if name in CLASSES:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                cands.append({"box": (x1, y1, x2, y2), "golongan_name": name, "conf": float(b.conf[0])})
    kept = filter_colliding_boxes(cands, overlap_threshold=OVERLAP)
    if not kept:
        return "TD", None
    best = max(kept, key=lambda d: d["conf"])
    return best["golongan_name"], best["conf"]


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", default=None, help="Folder dataset uji (default: test / Dataset_Uji)")
    ap.add_argument("--out_dir", default=None, help="Folder hasil uji (default: Hasil_Uji di folder script)")
    ap.add_argument("--device", default=None, help="cpu / 0 (GPU). Default: otomatis")
    ap.add_argument("--val", action="store_true", help="jalankan juga model.val() pada data.yaml")
    a = ap.parse_args()

    # Resolusi path test_dir secara absolut & otomatis mendeteksi folder script
    if a.test_dir:
        if os.path.isabs(a.test_dir) and os.path.exists(a.test_dir):
            test_dir = a.test_dir
        elif os.path.exists(os.path.join(BASE_DIR, a.test_dir)):
            test_dir = os.path.join(BASE_DIR, a.test_dir)
        elif os.path.exists(os.path.abspath(a.test_dir)):
            test_dir = os.path.abspath(a.test_dir)
        else:
            test_dir = os.path.join(BASE_DIR, a.test_dir)
    else:
        # Prioritaskan folder test/ jika ada, fallback ke Dataset_Uji/
        test_path = os.path.join(BASE_DIR, "test")
        dataset_uji_path = os.path.join(BASE_DIR, "Dataset_Uji")
        if os.path.exists(test_path) and (os.path.exists(os.path.join(test_path, "images")) or any(os.path.exists(os.path.join(test_path, g)) for g in CLASSES)):
            test_dir = test_path
        elif os.path.exists(dataset_uji_path):
            test_dir = dataset_uji_path
        else:
            test_dir = test_path

    # Resolusi path out_dir secara absolut
    if a.out_dir:
        out_dir = a.out_dir if os.path.isabs(a.out_dir) else os.path.join(BASE_DIR, a.out_dir)
    else:
        out_dir = os.path.join(BASE_DIR, "Hasil_Uji")

    os.makedirs(out_dir, exist_ok=True)
    print(f"[+] Direktori Dataset Uji : {test_dir}")
    print(f"[+] Direktori Output Hasil: {out_dir}")

    detector = get_detector_model()
    device = a.device if a.device is not None else (0 if __import__("torch").cuda.is_available() else "cpu")
    if isinstance(device, str) and device.isdigit():
        device = int(device)
    print(f"[+] Device: {device}")

    items = []
    # 1. Format Subfolder per golongan (e.g. test/Golongan I atau Dataset_Uji/Golongan I)
    has_subfolders = any(os.path.isdir(os.path.join(test_dir, gol)) for gol in CLASSES)
    if has_subfolders:
        for gol in CLASSES:
            d = os.path.join(test_dir, gol)
            files = sorted(f for f in os.listdir(d) if f.lower().endswith(EXTS)) if os.path.isdir(d) else []
            if len(files) != 10:
                print(f"[!] {gol}: ditemukan {len(files)} citra (seharusnya 10)")
            items += [(gol, os.path.join(d, f), f) for f in files]
    else:
        # 2. Format YOLO (test/images + test/labels seperti Dataset_Original/test)
        img_dir = os.path.join(test_dir, "images") if os.path.exists(os.path.join(test_dir, "images")) else test_dir
        lbl_dir = os.path.join(test_dir, "labels")
        if os.path.exists(img_dir) and os.path.exists(lbl_dir):
            for fname in sorted(os.listdir(img_dir)):
                if fname.lower().endswith(EXTS):
                    lbl_p = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
                    if os.path.exists(lbl_p):
                        best_area = -1.0
                        best_cid = None
                        with open(lbl_p, "r", encoding="utf-8") as lf:
                            for line in lf:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    cid = int(parts[0])
                                    w, h = float(parts[3]), float(parts[4])
                                    area = w * h
                                    if area > best_area:
                                        best_area = area
                                        best_cid = cid
                        if best_cid is not None and 0 <= best_cid < len(CLASSES):
                            items.append((CLASSES[best_cid], os.path.join(img_dir, fname), fname))
        print(f"[+] Terdeteksi format YOLO: {len(items)} citra dengan ground-truth label.")

    if not items:
        raise SystemExit(f"Tidak ada citra uji ditemukan di: {test_dir}")

    # pemanasan 5 citra
    for _, p, _ in items[:5]:
        img = load_image(p)
        if img is not None:
            predict_image(detector, img, device)

    labels = CLASSES + ["TD"]
    cm = np.zeros((6, 7), dtype=int)
    rows, times = [], []
    for i, (gol, path, fname) in enumerate(items, 1):
        img = load_image(path)
        if img is None:
            print(f"[!] Gagal membaca {path}")
            continue
        t0 = time.perf_counter()
        pred, conf = predict_image(detector, img, device)
        dt = time.perf_counter() - t0
        times.append(dt)
        cm[CLASSES.index(gol), labels.index(pred)] += 1
        rows.append([i, gol, fname, pred, "" if conf is None else round(conf * 100, 1),
                     "Benar" if pred == gol else "Salah", round(dt * 1000, 1)])

    with open(os.path.join(out_dir, "log_pengujian.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["No", "Aktual", "Berkas", "Prediksi", "Keyakinan(%)", "Hasil", "Waktu(ms)"])
        w.writerows(rows)

    per, tot_ok, tot_n = {}, 0, 0
    for i, g in enumerate(CLASSES):
        tp = int(cm[i, i]); n = int(cm[i].sum())
        fp = int(cm[:, i].sum() - tp); fn = n - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / n if n else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        lo, hi = wilson(tp, n)
        confs = [r[4] for r in rows if r[1] == g and r[4] != ""]
        per[g] = dict(n=n, benar=tp, salah=n - tp, akurasi=round(100 * rec, 1),
                      ci_low=round(100 * lo, 1), ci_high=round(100 * hi, 1),
                      precision=round(prec, 3), recall=round(rec, 3), f1=round(f1, 3),
                      conf_mean=round(float(np.mean(confs)), 1) if confs else None,
                      conf_min=min(confs) if confs else None, conf_max=max(confs) if confs else None)
        tot_ok += tp; tot_n += n
    mean_t = float(np.mean(times)) if times else 0.0
    out = dict(device=str(device), total=tot_n, benar=tot_ok,
               akurasi_total=round(100 * tot_ok / tot_n, 1) if tot_n else None,
               macro_precision=round(float(np.mean([v["precision"] for v in per.values()])), 3),
               macro_recall=round(float(np.mean([v["recall"] for v in per.values()])), 3),
               macro_f1=round(float(np.mean([v["f1"] for v in per.values()])), 3),
               ms_per_image=round(mean_t * 1000, 1), fps=round(1 / mean_t, 1) if mean_t else None,
               confusion_labels=labels, confusion=cm.tolist(), per_golongan=per)

    if a.val:
        try:
            data_yaml = os.path.join(BASE_DIR, "data.yaml")
            m = detector.val(data=data_yaml, device=device, verbose=False)
            names = detector.names
            val = {}
            for k, ci in enumerate(m.box.ap_class_index):
                val[str(names[int(ci)])] = dict(precision=round(float(m.box.p[k]), 3), recall=round(float(m.box.r[k]), 3),
                                                map50=round(float(m.box.ap50[k]), 3), map50_95=round(float(m.box.ap[k]), 3))
            val["_rata2"] = dict(precision=round(float(m.box.mp), 3), recall=round(float(m.box.mr), 3),
                                 map50=round(float(m.box.map50), 3), map50_95=round(float(m.box.map), 3))
            out["validasi"] = val
        except Exception as e:
            print(f"[!] model.val() gagal: {e}")

    with open(os.path.join(out_dir, "hasil_uji.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    lines = [f"Device: {device} | Total benar: {tot_ok}/{tot_n} ({out['akurasi_total']}%)",
             f"Macro P/R/F1: {out['macro_precision']}/{out['macro_recall']}/{out['macro_f1']}",
             f"Waktu rata-rata: {out['ms_per_image']} ms/citra ({out['fps']} FPS)", ""]
    for g, v in per.items():
        lines.append(f"{g:<13} {v['benar']}/{v['n']}  akurasi {v['akurasi']}%  CI95 [{v['ci_low']}, {v['ci_high']}]  "
                     f"P {v['precision']} R {v['recall']} F1 {v['f1']}")
    lines += ["", "Confusion matrix (baris=aktual, kolom=" + ", ".join(labels) + "):"] + [str(r) for r in cm.tolist()]
    txt = "\n".join(lines)
    open(os.path.join(out_dir, "ringkasan.txt"), "w", encoding="utf-8").write(txt)
    print(txt)
    print(f"\n[v] Selesai. Hasil di folder '{out_dir}/'.")


if __name__ == "__main__":
    main()
