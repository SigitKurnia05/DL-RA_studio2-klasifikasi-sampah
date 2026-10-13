# Catatan kerja — Studio 2 (Preview 1)

Buat Bang Romadhon. Ini isi kepala saya soal repo ini, biar nggak bingung waktu buka.

## 1. Kondisi sekarang

Yang sudah ada di repo:

- `splits/val.txt` — daftar validasi beku kita: 413 gambar dari 2.753, seed 42, hash kanonik `36bee95e...`. **Jangan dibikin ulang.** Semua eksperimen harus dinilai di daftar ini biar angkanya sebanding.
- `train.py` — skrip latih. Val-nya dibaca dari `splits/val.txt` (nggak pernah bikin split sendiri). Checkpoint terbaik dipilih pakai **macro-F1** dari fungsi `metrics()` di `evaluate.py`, bukan akurasi. Argumennya: `--data --split --model --out --epochs --bs --lr --wd --aug --history-out --workers --limit`.
- `make_split.py` — bikin/periksa split beku. Kalau datasetnya nggak berubah, cukup `--check` (nggak nulis apa pun).
- `submission/model.py` — **EfficientNetV2-S dari torchvision**, crop 288 (resize 320 di val). Satu-satunya tempat angka crop didefinisikan (`CROP`, `RESIZE`, `IMAGENET_WEIGHTS`).
- `notebooks/train_colab.ipynb` — runner Colab. Nama eksperimen diambil dari `EXP_NAME`, jadi bobot/grafik/riwayat tiap eksperimen punya berkas sendiri.
- `evaluate.py` — skrip penilai resmi, sama persis dengan punya asisten.

Yang **belum** dan bikin repo belum bisa dinilai:

- `submission/model.pth` masih **0 byte**. Wajib diisi, kalau nggak asisten nggak dapet apa-apa buat dijalankan.
- README masih template: belum ada ringkasan pendekatan, cara reproduksi, dan tabel hasil. Kolom NIM sudah ada.

## 2. Ritual tiap mau latih

```bash
# 1. dataset ada di luar repo (jangan pernah taruh di dalam repo)
#    D:\...\IF25-40401 Deep Learning\dataset-studio2\train

# 2. split beku masih valid? (nggak nulis apa pun)
python make_split.py --check --data <dataset>/train

# 3. latih
python train.py --data <dataset>/train --epochs 12 --bs 32 --lr 1.5e-4 --wd 1e-2 --aug strong \
    --out submission/model.pth --history-out runs/exp2_history.json

# 4. uji pakai penilai resmi SEBELUM commit
python evaluate.py --submission ./submission --data ./my_val
```

`my_val/` dibuat otomatis oleh `train.py` (413 gambar) dari daftar beku. Kalau folder itu sudah ada, isinya nggak ditimpa — jadi angkanya tetap dari daftar yang sama.

Di laptop ini latihan itu berat (CPU dua core). Yang dipakai sehari-hari: latih di Colab, bobotnya turun ke `submission/model.pth`, uji di laptop.

## 3. Aturan main yang nggak boleh dilanggar

1. **Jangan commit dataset.** 2.753 gambar + `train_sources.csv` di luar repo. Repo ini publik dan aturan lomba melarang unggah ulang dataset.
2. **Jangan commit `splits/val.txt` versi baru.** Sekali angkanya berubah, semua hasil lama nggak sebanding lagi.
3. **`model.py` dan `model.pth` satu paket.** `evaluate.py` memuat keduanya dengan `strict=True`. Salah satu ketinggalan → gagal load → skor 0 di putaran itu. Jadi promosi = **satu commit, dua berkas**.
4. **Jangan bikin `.py` sekali pakai di repo.** Yang resmi cuma `train.py`, `make_split.py`, `evaluate.py`, `submission/model.py`, dan notebook Colab. Eksperimen sementara taruh di luar repo (atau di `ekperiment/`, lihat catatan di bawah).
5. **Jangan modif `evaluate.py`.** Itu salinan skrip asisten; biar angkanya identik.
6. `submission/requirements.txt` isinya paket tambahan buat asisten. Sekarang torch/torchvision/numpy/pillow — semuanya paket dasar yang sudah ada di environment mereka. Kalau ada paket baru yang benar-benar dipakai, tulis di situ.

## 4. Tanda-tanda overfitting

Gejala: `train_loss` terus turun tapi `val_loss` mulai naik (di eksperimen pertama naik tajam dari epoch 6). `macro_f1` bisa masih naik sebentar setelahnya, jadi **jangan cuma lihat macro_f1**.

Kalau kejadian, urutan rem yang paling murah dulu:
1. `--aug strong` (RRC 0.5-1.0 + vflip + RandAugment(2, 9) + RandomErasing 0.25).
2. Turunin `--lr` (mis. 1.5e-4 → 1e-4).
3. Naikin `--wd` (1e-2 → 3e-2).
4. `--epochs` dikurangin + ambil checkpoint terbaik saja (train.py sudah nyimpen bobot terbaik tiap ada rekor).
5. Baru setelah itu pikirkan yang mahal: unfreeze bertahap, atau LR beda antara backbone dan kepala.

## 5. Eksperimen yang bisa dicoba buat Preview 1

Bahan dijalankan satu-satu, jangan borong. Pembanding tetap: **EfficientNetV2-S, lr 1.5e-4, wd 1e-2, aug strong, 12 epoch**.

| # | Yang dicoba | Perintah singkat | Yang diamati |
|---|---|---|---|
| 1 | Jadwal epoch lebih panjang | `--epochs 20` (patience default 4) | val_loss mulai naik di epoch berapa |
| 2 | Augmentasi vs jadwal LR, satu-satu | `--aug strong --lr 1e-4` vs `--aug strong --epochs 20` | mana yang lebih ngaruh |
| 3 | Rem lebih kencang | `--wd 3e-2` | jarak train-val loss menyempit? |
| 4 | Kelas lemah | lihat F1 `kertas_kardus` di `evaluate.py` | itu kelas paling susah (cuma 365 gambar) |
| 5 | Bobot+EMA | belum ada di `train.py` | kalau mau coba, kabarin dulu |
| 6 | TTA cermin saat inferensi | ubah `model.py` (di val cuma mirror hflip) | tambah macro-f1 tipis, waktu naik ×2 |
| 7 | Fusi dua backbone | `model.py` + bobot dua cabang | paling mahal, barunya kalau di atas mentok |

Aturan penilaian eksperimen: **selisih di bawah ±0.01 macro-f1 belum bisa dipercaya**, karena 413 gambar berarti satu gambar nilainya sekitar 0.0024. Naikin satu versi model cuma kalau bedanya jelas, atau arahnya konsisten (mis. tren memburuk di semua setelan).

## 6. Hasil yang sudah diukur

Semuanya dinilai dengan `evaluate.py` di 413 gambar beku yang sama, di laptop ini.

| Model | Macro-F1 | Akurasi | Salah | Catatan |
|---|---|---|---|---|
| EfficientNetV2-S @288, 3 epoch | 0.9555 | 0.9564 | 18 | val loss meledak mulai epoch 6 |
| EfficientNetV2-S @288, 12 epoch, aug strong, lr 1.5e-4, wd 1e-2 | 0.9590 | 0.9613 | 16 | bobot `exp2_effnet_v2s_reg.pth` |

Baseline asisten: ResNet-18 pretrained 8 epoch ≈ **0.92** di validation set tersembunyi (±740 gambar, split beda), jadi angka di tabel ini nggak bisa dibandingin langsung sama 0.92.

F1 per kelas model exp2: botol 0.9747 · gelas 0.9677 · kertas_kardus 0.9273 · saset 0.9600 · styrofoam 0.9655. Kertas_kardus tetap yang terlemah.

## 7. Tempat eksperimen

- `runs/` — sudah masuk `.gitignore`, aman buat riwayat JSON dan checkpoint mentah.
- `eksperiment/` — ada **di dalam** folder repo ini, isinya arsip bobot + grafik eksperimen (`model_effnet_v2s.pth`, `training_curves.png`, `exp2_effnet_v2s_reg.pth`, `exp2_effnet_v2s_reg_curves.png`).
  - Bobotnya nggak ke-commit karena `.gitignore` punya pola `*.pth`. Grafik `.png`-nya **belum masuk pola ignore apa pun**, jadi kalau nanti `git add .` dijalankan, empat berkas itu ikut ter-commit dan repo jadi berat (±390 KB, nggak fatal tapi kotor).
  - Ini masih folder baru yang belum diputuskan: mau dijadikan tempat arsip eksperimen yang resmi, atau di luar repo aja. Kalau mau diarsipkan di repo, grafiknya aman di-commit dan yang nggak boleh cuma bobotnya (sudah di-ignore). Kalau nggak, hapus aja sebelum push.

jika sudah selesai dengan eksperimen atau skema baru yang ada silahkan hapus permanen catatan ini sebelum di push, oh iya satu lagi google colab nya masih terlalu siingkat gitu kurang informatif silahkan dikembangkan yaa