# -*- coding: utf-8 -*-
"""
cek_kebocoran.py - memeriksa apakah citra uji identik / mirip dengan citra latih atau validasi.
Letakkan di folder proyek (sejajar Dataset_Uji, train, valid), lalu jalankan:
    python cek_kebocoran.py
    python cek_kebocoran.py --ambang 8      # lebih longgar (default 6 dari 64 bit)
Keluaran: layar + Hasil_Uji/cek_kebocoran.csv
"""
import os, sys, csv, hashlib, argparse
import cv2, numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif")

def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()

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

def dhash(img, n=8):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.resize(g, (n + 1, n), interpolation=cv2.INTER_AREA)
    return (g[:, 1:] > g[:, :-1]).flatten()

def resolve_dir(path):
    if not path:
        return None
    if os.path.isabs(path) and os.path.exists(path):
        return path
    cand_base = os.path.join(BASE_DIR, path)
    if os.path.exists(cand_base):
        return cand_base
    cand_cwd = os.path.abspath(path)
    if os.path.exists(cand_cwd):
        return cand_cwd
    return cand_base

def kumpul(root):
    out = []
    if not root or not os.path.isdir(root):
        return out
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(EXTS):
                out.append(os.path.join(r, f))
    return out

def main():
    default_uji = "test" if os.path.exists(os.path.join(BASE_DIR, "test")) else "Dataset_Uji"
    ap = argparse.ArgumentParser()
    ap.add_argument("--uji", default=default_uji, help=f"Folder dataset uji (default: {default_uji})")
    ap.add_argument("--ref", nargs="+", default=["train", "valid"], help="Folder pembanding (default: train valid)")
    ap.add_argument("--ambang", type=int, default=6)
    ap.add_argument("--out_dir", default=None, help="Folder output hasil (default: Hasil_Uji)")
    a = ap.parse_args()

    uji_dir = resolve_dir(a.uji)
    ref_dirs = [resolve_dir(d) for d in a.ref]
    out_dir = resolve_dir(a.out_dir) if a.out_dir else os.path.join(BASE_DIR, "Hasil_Uji")

    print(f"[+] Folder uji        : {uji_dir}")
    print(f"[+] Folder pembanding : {ref_dirs}")
    print(f"[+] Folder output     : {out_dir}")

    uji = kumpul(uji_dir)
    ref = [p for d in ref_dirs if os.path.isdir(d) for p in kumpul(d)]
    print(f"Citra uji: {len(uji)} | citra pembanding (latih/validasi): {len(ref)}")
    if not uji or not ref:
        sys.exit(f"Folder tidak ditemukan atau kosong. Cek --uji ({uji_dir}) dan --ref ({ref_dirs}).")

    rmd5, rh = {}, []
    for p in ref:
        im = load_image(p)
        if im is None: continue
        rmd5.setdefault(md5(p), p)
        rh.append((p, dhash(im)))
    H = np.array([h for _, h in rh])

    rows, n_ident, n_mirip = [], 0, 0
    for p in sorted(uji):
        im = load_image(p)
        if im is None: continue
        status, cocok, jarak = "BERSIH", "", ""
        m = md5(p)
        if m in rmd5:
            status, cocok, jarak = "IDENTIK", rmd5[m], 0
            n_ident += 1
        else:
            h = dhash(im)
            d = (H != h).sum(axis=1)
            k = int(d.argmin())
            # cek juga versi cermin (augmentasi flip)
            d2 = (H != dhash(cv2.flip(im, 1))).sum(axis=1)
            k2 = int(d2.argmin())
            if d2[k2] < d[k]: k, d = k2, d2
            if d[k] <= a.ambang:
                status, cocok, jarak = "MIRIP", rh[k][0], int(d[k])
                n_mirip += 1
            else:
                jarak = int(d[k])
        rel_path = os.path.relpath(p, uji_dir)
        rows.append([rel_path, status, jarak, cocok])
        if status != "BERSIH":
            print(f"[{status}] {rel_path}  <->  {cocok}  (jarak {jarak})")

    bersih = len(rows) - n_ident - n_mirip
    print(f"\nRINGKASAN: identik {n_ident}, mirip {n_mirip}, bersih {bersih} dari {len(rows)} citra uji.")

    # Simpan hasil CSV dengan penanganan error izin aman
    try:
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, "cek_kebocoran.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Citra uji", "Status", "Jarak dHash", "Pembanding"])
            w.writerows(rows)
        print(f"Detail: {out_csv}")
    except (PermissionError, OSError) as e:
        fallback_csv = os.path.abspath("cek_kebocoran.csv")
        try:
            with open(fallback_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Citra uji", "Status", "Jarak dHash", "Pembanding"])
                w.writerows(rows)
            print(f"Detail (disimpan di lokasi saat ini): {fallback_csv}")
        except Exception:
            print(f"[!] Catatan: Tidak dapat menulis file CSV ({e}), seluruh ringkasan tetap tercetak di atas.")

if __name__ == "__main__":
    main()
