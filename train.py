"""Latih submission pada split validasi beku splits/val.txt.

Skrip ini TIDAK pernah membuat split baru. Daftar validasi dibaca dari berkas yang
sudah dibekukan (splits/val.txt) supaya hasil dari laptop, Colab, dan tiap anggota
selalu dinilai pada gambar yang sama.

Keputusan penting:
  - Checkpoint terbaik dipilih memakai MACRO-F1, bukan akurasi. Metrik dan urutan
    kelasnya diambil langsung dari evaluate.py, jadi angka yang kita lihat identik
    dengan yang dihitung penilai.
  - Arsitektur dan transform validasi diambil dari berkas yang ditunjuk --model
    (default submission/model.py) supaya berkas submission tetap satu sumber
    kebenaran; model.py dan model.pth-nya tidak bisa berbeda diam-diam.
  - Augmentasi dan optimasi dibuat standar dan stabil dulu. Mixup dan EMA sengaja
    belum dipakai (dijadwalkan untuk putaran berikutnya).
  - Bobot bobot kandidat disimpan ke --out hanya kalau macro-F1-nya membaik; salinan
    kerja disimpan di runs/best.pth supaya proses panjang tidak hilang kalau putus.

Pemakaian:
    python train.py --data <dataset>/train                    # latih -> submission/model.pth
    python train.py --data <dataset>/train --limit 24 --epochs 1   # uji cepat (smoke test)
    python train.py --data <dataset>/train --model submission/model.py --out runs/cand.pth

Argumen penting:
    --data: folder dataset berisi <kelas>/*.jpg.
    --split: daftar validasi beku (default splits/val.txt).
    --model: berkas berisi get_model()/get_transform() (default submission/model.py).
    --out: tujuan bobot terbaik (default submission/model.pth).
    --limit: batasi jumlah gambar latih untuk uji cepat; 0 = semua (default 0).
    --scratch: latih tanpa bobot ImageNet.

Returns:
    None; menulis bobot terbaik dan runs/last.json berisi riwayat tiap epoch.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from evaluate import CLASSES, metrics  # noqa: E402  (urutan kelas & metrik milik penilai)


def load_model_module(path):
    """Impor berkas model (get_model/get_transform) dari path apa pun.

    Folder berkasnya ikut masuk sys.path supaya model.py boleh mengimpor berkas
    pendukung di sebelahnya, sama seperti yang dilakukan evaluate.py.

    Args:
        path: Path ke berkas model.

    Returns:
        module: modul yang sudah dimuat.

    Raises:
        SystemExit: bila berkas tidak ada atau tidak mendefinisikan fungsi wajib.
    """
    p = Path(path).resolve()
    if not p.is_file():
        raise SystemExit(f"[GAGAL] berkas model tidak ditemukan: {p}")
    sys.path.insert(0, str(p.parent))
    spec = importlib.util.spec_from_file_location("model_candidate", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for fn in ("get_model", "get_transform"):
        if not hasattr(mod, fn):
            raise SystemExit(f"[GAGAL] {p} tidak mendefinisikan {fn}()")
    return mod


class SplitList(Dataset):
    """Dataset dari daftar path relatif 'kelas/nama.jpg' (isi splits/val.txt)."""

    def __init__(self, root, names, tf):
        self.root, self.names, self.tf = Path(root), names, tf

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        rel = self.names[i]
        return self.tf(Image.open(self.root / rel).convert("RGB")), CLASSES.index(rel.split("/")[0])


def load_pretrained(model, mod):
    """Salin bobot ImageNet ke seluruh layer kecuali kepala (kepala tetap acak).

    Sumber bobot diambil dari penunjuk IMAGENET_WEIGHTS milik berkas model. Kalau tidak
    ada, sumbernya ditebak lewat torchvision. Tensor yang bentuknya berbeda (mis. kepala
    1000 kelas vs 5 kelas pada arsitektur lain) otomatis dilewati.

    Args:
        model: torch.nn.Module yang sudah dibangun (kepala 5 kelas).
        mod: modul model yang sedang dilatih (menyediakan IMAGENET_WEIGHTS bila ada).

    Returns:
        None; bobot model dimutasi langsung.

    Raises:
        SystemExit: bila bobot ImageNet tidak dapat ditemukan / tidak cocok.
    """
    weights = getattr(mod, "IMAGENET_WEIGHTS", None)
    if weights is None:
        try:
            enum = models.get_model_weights(model)
            weights = getattr(enum, "IMAGENET1K_V1", None) or enum.DEFAULT
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                f"[GAGAL] tidak tahu sumber bobot ImageNet untuk {type(model).__name__}: {exc}\n"
                f"        tambahkan IMAGENET_WEIGHTS di berkas model.")
    sumber = weights.get_state_dict(progress=True)
    sendiri = model.state_dict()
    cocok = {k: v for k, v in sumber.items() if k in sendiri and sendiri[k].shape == v.shape}
    dilewati = [k for k in sumber if k not in cocok]
    missing, unexpected = model.load_state_dict(cocok, strict=False)
    if unexpected:
        raise SystemExit(f"[GAGAL] bobot ImageNet tidak cocok dengan arsitektur: {unexpected[:3]}")
    sisa = [k for k in missing if k not in dilewati]
    if sisa:
        raise SystemExit(f"[GAGAL] layer tidak terisi bobot: {sisa[:3]}")
    print(f"pretrained: {weights} | {len(cocok)} tensor dimuat | {len(dilewati)} tensor kepala dilewati")


def train_epoch(model, dl, opt, lossf, device):
    """Satu putaran pelatihan.

    Args:
        model: model yang dilatih.
        dl: DataLoader data latih.
        opt: optimizer.
        lossf: fungsi kerugian.
        device: 'cuda' atau 'cpu'.

    Returns:
        float: rata-rata kerugian sepanjang epoch.
    """
    model.train()
    total = 0.0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = lossf(model(x), y)
        loss.backward()
        opt.step()
        total += loss.item()
    return total / max(1, len(dl))


@torch.no_grad()
def hitung_loss(model, dl, lossf, device):
    """Rata-rata kerugian pada sebuah DataLoader (tanpa gradien).

    Args:
        model: model yang dinilai.
        dl: DataLoader.
        lossf: fungsi kerugian yang sama dengan saat pelatihan.
        device: 'cuda' atau 'cpu'.

    Returns:
        float: rata-rata kerugian berbobot jumlah gambar.
    """
    model.eval()
    total, n = 0.0, 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        total += lossf(model(x), y).item() * y.size(0)
        n += y.size(0)
    return total / max(1, n)


@torch.no_grad()
def evaluate_on(model, dl, device):
    """Hitung metrik penilai pada sebuah DataLoader.

    Args:
        model: model yang dinilai.
        dl: DataLoader data validasi.
        device: 'cuda' atau 'cpu'.

    Returns:
        tuple: (hasil_metrik, y_true, y_pred) dengan hasil_metrik berisi macro_f1,
        accuracy, per_class_f1, dan confusion_matrix.
    """
    model.eval()
    y_true, y_pred = [], []
    for x, y in dl:
        y_pred += model(x.to(device)).argmax(1).cpu().tolist()
        y_true += y.tolist()
    return metrics(y_true, y_pred), y_true, y_pred


def salin_val(data, names, tujuan):
    """Salin gambar validasi ke my_val/ agar bisa diuji evaluate.py.

    Args:
        data: folder dataset sumber.
        names: daftar path relatif hasil split.
        tujuan: folder tujuan.

    Returns:
        int: jumlah berkas yang disalin (0 bila folder sudah ada).
    """
    tujuan = Path(tujuan)
    if tujuan.exists():
        return 0
    for rel in names:
        keluar = tujuan / rel
        keluar.parent.mkdir(parents=True, exist_ok=True)
        keluar.write_bytes((Path(data) / rel).read_bytes())
    return len(names)


def main():
    """Titik masuk CLI pelatihan."""
    ap = argparse.ArgumentParser(description="Latih submission Studio 2 pada split beku.")
    ap.add_argument("--data", required=True, help="folder dataset berisi <kelas>/*.jpg")
    ap.add_argument("--split", default="splits/val.txt", help="daftar validasi beku")
    ap.add_argument("--model", default="submission/model.py", help="berkas get_model()/get_transform()")
    ap.add_argument("--out", default="submission/model.pth", help="tujuan bobot terbaik")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=4, help="berhenti bila macro-F1 val tak naik N epoch")
    ap.add_argument("--limit", type=int, default=0, help="batasi gambar latih (uji cepat); 0 = semua")
    ap.add_argument("--scratch", action="store_true", help="latih tanpa bobot ImageNet")
    a = ap.parse_args()

    mod = load_model_module(a.model)
    get_model, get_transform = mod.get_model, mod.get_transform
    crop = getattr(mod, "CROP", None)
    print(f"model file: {Path(a.model).resolve()}" + (f" | crop={crop}" if crop else ""))

    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_names = [b.strip() for b in Path(a.split).read_text(encoding="utf-8").splitlines() if b.strip()]
    semua = [f"{c}/{p.name}" for c in CLASSES for p in sorted((Path(a.data) / c).glob("*.jpg"))]
    kunci_val = set(val_names)
    tr_names = [n for n in semua if n not in kunci_val]
    if not tr_names or not val_names:
        raise SystemExit(f"[GAGAL] data latih/validasi kosong (train={len(tr_names)} val={len(val_names)})")
    if a.limit:
        tr_names = tr_names[:a.limit]
    print(f"device={device} | train={len(tr_names)} val={len(val_names)} | split={a.split}")

    sisi = crop or 224
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(sisi, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tr_dl = DataLoader(SplitList(a.data, tr_names, train_tf), batch_size=a.bs, shuffle=True,
                       num_workers=a.workers, drop_last=len(tr_names) > a.bs,
                       persistent_workers=a.workers > 0)
    va_dl = DataLoader(SplitList(a.data, val_names, get_transform()), batch_size=a.bs,
                       num_workers=a.workers)
    print(f"train: {len(tr_dl)} batch/epoch | aug: RRC({sisi}, 0.7-1.0)+hflip+ColorJitter")

    model = get_model().to(device)
    if not a.scratch:
        load_pretrained(model, mod)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)

    Path("runs").mkdir(exist_ok=True)
    best_f1, best_state, bad, riwayat = -1.0, None, 0, []
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, tr_dl, opt, lossf, device)
        lr_kini = opt.param_groups[0]["lr"]
        sched.step()  # jadwal melangkah setelah epoch selesai
        val_loss = hitung_loss(model, va_dl, lossf, device)
        m, _, _ = evaluate_on(model, va_dl, device)
        durasi = time.time() - t0
        rekor = m["macro_f1"] > best_f1
        tanda = " bintang rekor" if rekor else ""
        riwayat.append({"epoch": ep, "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4),
                        "val_macro_f1": round(m["macro_f1"], 4), "val_acc": round(m["accuracy"], 4),
                        "lr": lr_kini, "detik": round(durasi, 1), "rekor": rekor})
        print(f"ep {ep:2d}/{a.epochs} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"| val_macro_f1={m['macro_f1']:.4f} val_acc={m['accuracy']:.4f} "
              f"| lr={lr_kini:.2e} | {durasi:.0f}s{tanda}", flush=True)
        Path("runs/history.json").write_text(json.dumps(riwayat, indent=2), encoding="utf-8")
        if rekor:
            best_f1, bad = m["macro_f1"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, "runs/best.pth")  # simpan tiap perbaikan: proses panjang gampang putus
        else:
            bad += 1
            if bad >= a.patience:
                print(f"early stop: macro-F1 val tidak naik {a.patience} epoch")
                break

    model.load_state_dict(best_state)
    m, _, _ = evaluate_on(model, va_dl, device)
    print(f"\nTERBAIK val_macro_f1={m['macro_f1']:.4f} acc={m['accuracy']:.4f}")
    for c, f1 in m["per_class_f1"].items():
        print(f"  {c:16s} F1={f1:.4f}")
    print("confusion (baris=asli, kolom=prediksi):")
    print("      " + " ".join(f"{c[:7]:>7s}" for c in CLASSES))
    for c, row in zip(CLASSES, m["confusion_matrix"]):
        print(f"{c[:7]:>7s} " + " ".join(f"{v:7d}" for v in row))

    keluar = Path(a.out)
    keluar.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), keluar)
    print(f"\n-> {keluar} ({keluar.stat().st_size / 2**20:.1f} MB)")
    n = salin_val(a.data, val_names, "my_val")
    if n:
        print(f"my_val/ disalin: {n} gambar")
    print("uji: python evaluate.py --submission ./submission --data ./my_val")
    Path("runs/last.json").write_text(json.dumps(
        {"argv": vars(a), "best_macro_f1": round(best_f1, 4), "per_class_f1": m["per_class_f1"],
         "confusion_matrix": m["confusion_matrix"], "riwayat": riwayat}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
