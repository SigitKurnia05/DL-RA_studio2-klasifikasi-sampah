# Studio 2 — Kompetisi Klasifikasi Sampah

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

**Nama kelompok:** 11 (YOLO)

Repositori: https://github.com/SigitKurnia05/DL-RA_studio2-klasifikasi-sampah

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
| 1  | Muhammad Romadhon Santoso | 123140031 | *(menunggu konfirmasi anggota)* |
| 2  | Sigit Kurnia Hartawan | 123140033 | *(menunggu konfirmasi anggota)* |

## Ringkasan Pendekatan

| Nama | Penjelasan |
|---|---|
| **Arsitektur / backbone** | **ConvNeXt-Tiny** pretrained ImageNet-1K (`torchvision.models.convnext_tiny`) |
| **Strategi pelatihan** | Full fine-tuning (seluruh layer dilatih); AdamW + CosineAnnealingLR; maksimal 20 epoch; early stopping; bobot terbaik dipilih berdasarkan **macro-F1 validasi** |
| **Augmentasi** | RandomResizedCrop(0,5–1,0), flip vertikal, RandAugment(2, 9), RandomErasing p=0,25, normalisasi ImageNet |
| **Penanganan kelas tidak seimbang** | Belum diterapkan; `CrossEntropyLoss` dengan `label_smoothing=0,1`. Rencana perbaikan: class-weighted loss atau oversampling kelas minoritas |
| **Label noise** | `label_smoothing=0,1` |
| **Data tambahan** | **Tidak ada** hanya train set dari starter kit; dataset tidak diunggah ulang |
| **Ukuran input & preprocessing** | `Resize(256)` → `CenterCrop(224)` → `ToTensor` → normalisasi ImageNet (`mean [0.485, 0.456, 0.406]`, `std [0.229, 0.224, 0.225]`) |
| **Paket** | `torch`, `torchvision`, `pillow`, `numpy` tanpa `timm` |
| **Ukuran `submission/model.pth`** | **53,1 MB** dalam fp16 (fp32 berukuran 106,2 MB dan ditolak GitHub). Batas ketentuan 500 MB |
| **Waktu evaluasi** | 413 gambar = **24,7 detik** pada CPU 2 inti, setara ±45 detik untuk ±740 gambar tersembunyi jauh di bawah batas 15 menit |

## Hasil

Kolom "lokal" adalah macro-F1 epoch terbaik pada split validasi lokal kelompok sendiri
(413 gambar, seed 42). Kolom "asisten" adalah skor pada validation set tersembunyi yang hanya
dipegang tim pengajar.

| Putaran | Commit | Backbone | Macro-F1 lokal | Akurasi lokal | Macro-F1 asisten |
|---|---|---|---|---|---|
| Preview 1 (14 Okt) | [riwayat `submission/model.py`](https://github.com/SigitKurnia05/DL-RA_studio2-klasifikasi-sampah/commits/main/submission/model.py) | ConvNeXt-Tiny | 0,9658 | 0,9685 | *(menunggu pengumuman)* |
| Preview 2 (21 Okt) | *(belum)* | — | — | — | — |
| Final (25 Okt) | *(belum)* | — | — | — | — |

Hash commit yang dinilai dapat diperiksa dengan:

```bash
git log -1 --format=%H -- submission/model.pth
```

Sumber angka: `runs/exp4_convnext/history.json` (riwayat per epoch), `runs/exp4_convnext/metrics.json`
(ringkasan hasil), dan keluaran resmi `python evaluate.py --submission ./submission --data ./my_val`
yang dikerjakan ulang persis seperti asisten.

## Perbandingan Percobaan

Empat percobaan dijalankan pada **split validasi beku yang sama** (413 gambar, seed 42, hash kanonik
`36bee95ec932553b...`), seluruhnya dipilih berdasarkan macro-F1.

| Percobaan | Backbone | Crop | Augmentasi | lr / wd | batch | Epoch | Epoch terbaik | Macro-F1 | Akurasi | Salah |
|---|---|---|---|---|---|---|---|---|---|---|
| eksperimen 1 | EfficientNetV2-S | 288 | standar | 3e-4 / 1e-4 | — | 12 | 3 | 0,9555 | 0,9564 | 18 |
| eksperimen 2 | EfficientNetV2-S | 288 | strong | 1,5e-4 / 1e-2 | 32 | 12 | 9 | 0,9590 | 0,9613 | 16 |
| eksperimen 3 | EfficientNetV2-S | 288 | strong | 1,5e-4 / 1e-2 | 32 | 11 | 7 | 0,9595 | 0,9613 | 16 |
| **eksperimen 4** | **ConvNeXt-Tiny** | **224** | strong | **1e-4 / 1e-2** | **16** | 20 → 9 | **5** | **0,9658** | **0,9685** | **13** |

ConvNeXt-Tiny dipilih (kenaikan 0,0063 dari rekor sebelumnya) dan dipasang ke `submission/`.
Analisis lengkap tiap percobaan beserta grafiknya ada di `notebooks/riwayat_eksperimen.ipynb`
dan `docs/ANALISIS_EKSPERIMEN.md`.

### Catatan metodologis

- **Angka lokal bersifat optimistis.** Seluruh angka pada tabel di atas adalah epoch terbaik pada
  himpunan yang juga dipakai memilih epoch, sehingga cenderung melebihkan performa. Angka tersebut
  dilaporkan sebagai "epoch terbaik pada validasi lokal", bukan perkiraan skor validation set
  tersembunyi. Pembanding yang sah adalah kolom asisten.
- **Ambang noise.** Split lokal berisi 413 gambar, sehingga satu gambar bernilai ±0,0024 poin
  macro-F1. Selisih di bawah ±0,01 belum dapat dipercaya sebagai perbaikan.
- **Uji pengulangan.** Eksperimen 3 adalah pengulangan konfigurasi eksperimen 2 di lingkungan lokal
  dan menghasilkan selisih 0,0005, yaitu noise. Karena itu eksperimen 2 dan 3 diperlakukan sebagai
  satu konfigurasi.
- **Epoch terbaik bukan epoch terakhir.** Rekor eksperimen 4 jatuh pada epoch 5 dari 9; memakai bobot
  epoch terakhir akan menurunkan skor. Karena itu `train.py` menyimpan bobot hanya pada saat
  macro-F1 membaik.

## Distribusi Dataset yang Dipakai

| Kelas | Jumlah gambar | Proporsi |
|---|---:|---:|
| botol_plastik | 523 | 19,0% |
| gelas_plastik | 521 | 18,9% |
| kertas_kardus | 365 | 13,3% |
| saset_kemasan | 673 | 24,4% |
| styrofoam | 671 | 24,4% |
| **Total** | **2.753** | **100%** |

Rasio kelas terbesar terhadap terkecil adalah 1,84×. `kertas_kardus` adalah kelas dengan jumlah
gambar paling sedikit **dan** F1 terendah pada seluruh percobaan (0,9333 pada eksperimen 4; 5 dari
55 gambar validasi tertukar ke kelas lain). Perbaikan berikutnya paling menguntungkan diarahkan ke
kelas ini, bukan dengan mengganti backbone lagi.

## Pembagian Data (Latih / Validasi)

Pembagian dihitung sekali dengan skema berstrata per kelas dan seed tetap 42, lalu dibekukan di
`splits/val.txt`. Berkas tersebut tidak pernah ditulis ulang; menjalankan
`python make_split.py --check --data <dataset>/train` akan menghitung ulang hash kanoniknya dan
membandingkannya dengan berkas.

| | |
|---|---|
| **Latih** | 2.340 gambar (85%) |
| **Validasi (`my_val/`)** | 413 gambar (15%) — botol_plastik 78 · gelas_plastik 78 · kertas_kardus 55 · saset_kemasan 101 · styrofoam 101 |

`my_val/` hanya pembanding internal antar-run kelompok sendiri. Daftar gambar ujinya berbeda dari
validation set tersembunyi asisten, sehingga angkanya **tidak boleh** diklaim sebagai perkiraan
skor asisten.

## Penyimpanan Bobot: fp32 106,2 MB → fp16 53,1 MB

ConvNeXt-Tiny dalam fp32 berukuran 106,2 MB, sedangkan GitHub menolak berkas di atas 100 MB.
Karena itu `submission/model.pth` disimpan dalam **fp16**. Berkas arsitektur dan `get_transform()`
tidak berubah, dan `load_state_dict` mengubah bobot kembali ke fp32 saat dimuat, sehingga
perhitungan tetap berjalan pada presisi penuh.

| Pemeriksaan | Hasil |
|---|---|
| Seluruh tensor berupa float (tanpa `num_batches_tracked`) | ya; 182 tensor, 27.823.973 parameter |
| Error relatif maksimum bobot | 4,880e-04, tepat 2^-11 (batas presisi fp16) |
| `load_state_dict(strict=True)` melalui `evaluate.py` | lolos, tanpa kunci `missing`/`unexpected` |
| Macro-F1 fp32 berbanding fp16 pada 413 gambar | **0,9658 berbanding 0,9658** (identik) |
| Confusion matrix | identik sel per sel; 0 prediksi berubah |

## Cara Reproduksi

Dijalankan dari **root repositori**. Dataset train set tidak di-commit; arahkan `--data` ke folder
`train/` hasil unduhan resmi.

```bash
pip install -r requirements.txt

# 1. Periksa split beku (tidak menulis apa pun)
python make_split.py --check --data "<...>/train"

# 2. Latih ulang konfigurasi submission (ConvNeXt-Tiny crop 224)
python train.py --data "<...>/train" --model experiments/convnext_tiny.py \
                --epochs 20 --bs 16 --lr 1e-4 --wd 1e-2 --aug strong --patience 5 \
                --out runs/exp4_convnext/model.pth \
                --history-out runs/exp4_convnext/history.json

# 3. Wajib: uji dengan skrip evaluasi resmi milik asisten
python evaluate.py --submission ./submission --data ./my_val
```

`train.py` juga menyalin gambar validasi ke `my_val/` secara otomatis pada akhir pelatihan, sehingga
langkah 3 dapat dijalankan langsung.

Untuk menjalankan ulang percobaan di Google Colab, tersedia dua notebook dengan alur yang sama:
`notebooks/percobaan_convnext.ipynb` (ConvNeXt-Tiny crop 224) dan
`notebooks/percobaan_effnetv2s.ipynb` (EfficientNetV2-S crop 288). Keduanya memuat seluruh alur:
pemeriksaan GPU, pengambilan repositori, penyiapan dataset dari Google Drive, pelatihan, evaluasi
resmi, penyusunan grafik, konversi fp16 bila ukuran bobot melewati 100 MB, dan promosi ke
`submission/`. Setiap notebook menyimpan konfigurasinya sendiri sehingga tidak ada kemungkinan
menjalankan arsitektur yang keliru.

Riwayat lengkap seluruh percobaan beserta grafik perbandingannya ada di
`notebooks/riwayat_eksperimen.ipynb`, dan dapat dijalankan ulang karena membaca berkas di `runs/`.

## Struktur Repositori

```
.
├── README.md                          # berkas ini
├── requirements.txt                   # dependensi pelatihan
├── evaluate.py                        # salinan skrip evaluasi resmi (tidak diubah)
├── make_split.py                      # pembekuan & pemeriksaan split validasi
├── train.py                           # skrip pelatihan
├── experiments/
│   ├── convnext_tiny.py               # kandidat ConvNeXt-Tiny (224) -> TERPILIH
│   ├── efficientnet_v2s.py            # kandidat EfficientNetV2-S (288)
│   └── README.md
├── notebooks/
│   ├── percobaan_convnext.ipynb       # menjalankan ulang percobaan ConvNeXt-Tiny di Colab
│   ├── percobaan_effnetv2s.ipynb      # menjalankan ulang percobaan EfficientNetV2-S di Colab
│   └── riwayat_eksperimen.ipynb       # riwayat & analisis semua percobaan
├── submission/
│   ├── model.py                       # get_model() ConvNeXt-Tiny + get_transform() 224
│   ├── model.pth                      # state_dict terbaik, fp16 (53,1 MB)
│   └── requirements.txt
├── splits/
│   └── val.txt                        # split validasi beku (413 gambar, seed 42)
├── docs/
│   ├── ANALISIS_EKSPERIMEN.md         # analisis naratif hasil percobaan
│   ├── riwayat_eksperimen.png         # grafik perbandingan percobaan
│   └── kurva/                         # arsip grafik kurva pelatihan
└── runs/<tag>/                        # bukti percobaan: history.json, metrics.json, curves.png
```

`data/`, `my_val/`, `train_sources.csv`, dan seluruh bobot di dalam `runs/` tidak di-commit
(lihat `.gitignore`), sesuai ketentuan agar dataset tidak diunggah ulang ke repositori publik.
Berkas bukti angka tiap percobaan (`runs/<tag>/*.json` dan `curves.png`) **ikut** di-commit agar
hasilnya dapat diperiksa ulang.

## Catatan Teknis Penting

- **Urutan kelas wajib alfabetis**, sama dengan `torchvision.ImageFolder` dan `evaluate.py`:
  `0 botol_plastik · 1 gelas_plastik · 2 kertas_kardus · 3 saset_kemasan · 4 styrofoam`.
- **`submission/model.py` hanya membangun arsitektur.** Berkas tersebut memakai `weights=None`,
  tanpa mengunduh bobot dan tanpa memuat `model.pth`; yang memuat adalah `evaluate.py` dengan
  `strict=True`.
- **Resolusi latih dan resolusi uji harus sama.** Melatih pada satu resolusi lalu mengevaluasi pada
  resolusi lain menurunkan akurasi tanpa pesan galat; karena itu angka `CROP` dan `RESIZE` hanya
  didefinisikan di berkas arsitektur.
- **`model.py` dan `model.pth` wajib berasal dari percobaan yang sama** dan di-commit dalam satu
  commit. Bila hanya salah satunya diperbarui, submission gagal dimuat dan tidak mendapat skor.
- **Batas teknis:** `model.pth` berukuran 53,1 MB (≤ 500 MB) dan evaluasi 413 gambar selesai dalam
  24,7 detik pada CPU (≤ 15 menit), sehingga tidak diperlukan GitHub Release maupun Git LFS.
- **Ukuran berkas di GitHub:** berkas di atas 100 MB ditolak, dan berkas di atas 50 MB memunculkan
  peringatan. Bobot dalam fp16 berada pada kisaran 53 MB sehingga tetap dapat di-commit langsung.

## Sumber Daya

- Halaman resmi Studio 2: https://informatika-itera.github.io/deep-learning-IF25-40401/studio-2/
- Starter kit resmi: `evaluate.py`, `submission/model.py`, `train_baseline.py`, `README_TEMPLATE.md`
- Materi Pertemuan 4 (CNN from scratch) serta Pertemuan 5 (CNN dan Variasinya)
- Dataset: train set resmi mata kuliah (citra dari internet untuk keperluan pendidikan
  non-komersial; sumber per gambar tercatat pada `train_sources.csv` dan tidak di-commit)
