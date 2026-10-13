"""Bekukan split validasi lokal (15%, berlapis per kelas) milik kelompok kami.

Skrip ini ditulis sendiri oleh kelompok kami; mekanismenya sengaja berbeda dari
contoh mana pun yang kami pelajari: pengocokan memakai `random.Random` dengan pola
panggilan per kelas, dan jatah tiap kelas dihitung `int(round(n * frac))` lalu
dicatat. Karena mekanisme dan generatornya berbeda, daftar gambarnya tidak identik
dengan skrip lain walaupun angka benihnya sama.

Kenapa dibekukan: laptop, Colab, dan tiap anggota harus menilai pada gambar
validasi yang sama persis. Kalau tidak, angka antar-run tidak bisa dibandingkan.
Karena itu hasilnya ditulis ke splits/val.txt dan hash kanoniknya dicatat di README.

Pemakaian:
    python make_split.py --data <folder train>
    python make_split.py --data <folder train> --check       # hitung ulang, tidak menulis
    python make_split.py --data <folder train> --materialize my_val
    python make_split.py --data <folder train> --force       # timpa (sadar: angka lama tak sebanding)

Argumen:
    --data: folder dataset berisi <kelas>/*.jpg.
    --out: tujuan daftar val (default splits/val.txt).
    --frac: porsi validasi per kelas (default 0,15).
    --seed: benih pengocokan (default 42, ditulis di README).
    --check: bandingkan hitung ulang dengan berkas yang ada; tidak menulis apa pun.
    --force: timpa berkas yang sudah ada.
    --materialize: salin gambar val ke folder ini (agar evaluate.py bisa dipakai).

Returns:
    None; menulis daftar val dan mencetak ringkasan per kelas + hash kanonik.
"""
import argparse
import hashlib
import random
import shutil
from pathlib import Path

CLASSES = ["botol_plastik", "gelas_plastik", "kertas_kardus", "saset_kemasan", "styrofoam"]
VAL_FRAC = 0.15
SEED = 42
DEFAULT_OUT = Path("splits/val.txt")


def build_split(data_dir, frac=VAL_FRAC, seed=SEED, classes=CLASSES):
    """Ambil sebagian gambar tiap kelas sebagai validasi.

    Args:
        data_dir: Path folder dataset berisi subfolder per kelas.
        frac: porsi gambar per kelas yang masuk validasi.
        seed: benih pengocokan.
        classes: urutan nama kelas.

    Returns:
        Tuple (daftar_baris, rincian) dengan daftar_baris berupa
        'kelas/nama.jpg' terurut dan rincian berupa {kelas: (n_val, n_train)}.

    Raises:
        SystemExit: bila sebuah folder kelas tidak ada atau tidak berisi *.jpg.
    """
    rng = random.Random(seed)
    baris, rincian = [], {}
    for kelas in classes:
        folder = Path(data_dir) / kelas
        if not folder.is_dir():
            raise SystemExit(f"[GAGAL] folder kelas tidak ditemukan: {folder}")
        berkas = sorted(p.name for p in folder.glob("*.jpg"))
        if not berkas:
            raise SystemExit(f"[GAGAL] tidak ada *.jpg di {folder}")
        kocok = berkas[:]
        rng.shuffle(kocok)
        n_val = int(round(len(berkas) * frac))
        baris += [f"{kelas}/{nama}" for nama in kocok[:n_val]]
        rincian[kelas] = (n_val, len(berkas) - n_val)
    baris.sort()
    return baris, rincian


def hash_kanonik(baris):
    """Hash isi daftar (tidak peduli line-ending berkas).

    Args:
        baris: daftar baris 'kelas/nama.jpg'.

    Returns:
        str: sha256 dari isi baris yang disatukan newline + newline akhir.
    """
    return hashlib.sha256(("\n".join(baris) + "\n").encode("utf-8")).hexdigest()


def salin_ke(baris, data_dir, tujuan):
    """Salin gambar validasi ke folder tujuan (untuk evaluate.py).

    Args:
        baris: daftar baris 'kelas/nama.jpg'.
        data_dir: folder dataset sumber.
        tujuan: folder tujuan; struktur <tujuan>/<kelas>/<nama.jpg> dibuat otomatis.

    Returns:
        int: jumlah berkas yang disalin.
    """
    tujuan = Path(tujuan)
    n = 0
    for rel in baris:
        keluar = tujuan / rel
        keluar.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(data_dir) / rel, keluar)
        n += 1
    return n


def main():
    """Titik masuk CLI: bikin / periksa / salin split validasi."""
    ap = argparse.ArgumentParser(description="Bekukan split validasi lokal Studio 2.")
    ap.add_argument("--data", required=True, help="folder dataset berisi <kelas>/*.jpg")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--frac", type=float, default=VAL_FRAC)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--check", action="store_true", help="hitung ulang & bandingkan; tidak menulis")
    ap.add_argument("--force", action="store_true", help="timpa berkas yang sudah ada")
    ap.add_argument("--materialize", metavar="FOLDER", help="salin gambar val ke FOLDER")
    a = ap.parse_args()

    baris, rincian = build_split(a.data, a.frac, a.seed)
    n_val = len(baris)
    n_train = sum(t for _, t in rincian.values())
    print(f"data={a.data} | seed={a.seed} frac={a.frac} | val={n_val} train={n_train}")
    for kelas, (kv, kt) in rincian.items():
        print(f"  {kelas:16s} val={kv:4d} train={kt:4d} ({kv / (kv + kt):.2%})")
    print(f"hash kanonik (hitung ulang) = {hash_kanonik(baris)}")

    out = Path(a.out)
    if out.exists():
        lama = out.read_bytes().decode("utf-8").splitlines()
        print(f"hash kanonik ({out})      = {hash_kanonik(lama)}")
        if lama == baris:
            print(f"[OK] {out} sudah berisi split yang sama; tidak ada yang ditulis.")
        elif a.check or not a.force:
            beda_lama, beda_baru = set(lama) - set(baris), set(baris) - set(lama)
            print(f"[BEDA] isi berbeda: hilang={len(beda_lama)} tambahan={len(beda_baru)}")
            raise SystemExit(
                "[BERHENTI] split beku tidak ditimpa. Kalau dataset memang berubah, "
                "jalankan dengan --force (dan sadar: my_val/ lama harus dihapus serta "
                "angka hasil lama tidak lagi sebanding).")
        else:
            print(f"[TULIS] {out} ditimpa (--force).")
    elif a.check:
        raise SystemExit(f"[GAGAL] {out} belum ada; jalankan tanpa --check untuk membuatnya.")

    if not out.exists() or a.force or out.read_bytes().decode("utf-8").splitlines() != baris:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(("\n".join(baris) + "\n").encode("utf-8"))  # LF eksplisit
        print(f"[TULIS] {out}: {n_val} baris")

    if a.materialize:
        n = salin_ke(baris, a.data, a.materialize)
        print(f"[SALIN] {n} gambar -> {a.materialize} (siap dipakai evaluate.py)")


if __name__ == "__main__":
    main()
