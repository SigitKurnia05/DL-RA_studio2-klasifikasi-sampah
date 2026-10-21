# Folder `experiments/`

Berisi berkas arsitektur kandidat (berkas `.py` yang mendefinisikan `get_model()` dan
`get_transform()`) yang dipakai saat bereksperimen. Berkas di sini **bukan** submission:
submission resmi tetap di `submission/model.py`.

Alasan pemisahan ini: `train.py --model <berkas>` membangun arsitektur dari berkas yang ditunjuk,
jadi bila eksperimen menunjuk `submission/model.py`, berkas submission ikut berubah setiap kali
backbone dicoba. Dengan folder ini, berkas submission hanya berubah saat memang dipromosikan.

## Resep pelatihan menempel pada berkas arsitektur

Selain `get_model()` dan `get_transform()`, berkas di sini boleh mendefinisikan resep pelatihan yang
dibaca `train.py`:

| Nama | Arti | Bila tidak ada |
|---|---|---|
| `CROP`, `RESIZE` | ukuran crop latih/uji dan sisi pendek sebelum center-crop | `CROP` bawaannya 224 |
| `IMAGENET_WEIGHTS` | penunjuk enum bobot ImageNet yang diunduh `train.py` | ditebak dari `torchvision` |
| `HEAD_EPOCHS` | epoch pemanasan kepala dengan backbone dibekukan | 0 (langsung fine-tuning penuh) |
| `HEAD_LR` | learning rate tahap pemanasan kepala | 1e-3 |
| `CLASS_WEIGHTED` | memakai bobot kelas pada `CrossEntropyLoss` | `False` |
| `WEIGHT_POWER` | peluruhan bobot kelas, `w_c = (N_total / (K*N_c)) ** WEIGHT_POWER` | 1.0 |

Alasannya: satu percobaan menjadi satu berkas. Tidak ada kemungkinan menjalankan arsitektur A dengan
resep percobaan B, dan angka hasil selalu bisa ditelusuri ke berkas yang benar.

## Daftar kandidat

| Berkas | Arsitektur | Resolusi | Resep | Status |
|---|---|---|---|---|
| `convnext_tiny.py` | ConvNeXt-Tiny (torchvision, ImageNet-1K) | crop 224, resize 256 | tanpa pemanasan kepala, tanpa bobot kelas | percobaan 4, macro-F1 0,9658 |
| `convnext_ck224.py` | ConvNeXt-Tiny (torchvision, ImageNet-1K) | crop 224, resize 236 | pemanasan kepala 3 epoch, bobot kelas penuh | kandidat putaran 2 |
| `convnext_ck320.py` | ConvNeXt-Tiny (torchvision, ImageNet-1K) | crop 320, resize 448 | sama dengan `convnext_ck224.py` | kandidat jalur resolusi |

Kandidat EfficientNetV2-S sebelumnya didefinisikan langsung di `submission/model.py` dan masih
tersimpan sebagai salinan di `runs/exp3_effnetv2s/model.py`.

Bobot hasil latih **tidak** diletakkan di folder ini — bobot ditulis ke `runs/<tag>/` lewat
argumen `--out` pada `train.py`.

## Kenapa `RESIZE` 236, bukan 256

Resep resmi torchvision untuk bobot ImageNet-1K ConvNeXt-Tiny adalah `crop_size=224, resize_size=236`.
Angka itu bisa diperiksa dari metanya sendiri:

```python
from torchvision import models
print(models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1.meta["recipe"])
transforms = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms()
print(transforms.crop_size, transforms.resize_size)
```

Alasan memakai 236 adalah **kesesuaian dengan resolusi latih**, bukan karena transformasi ujinya lebih
baik. Uji pemisahan pada bobot lama (crop 224, dilatih dengan resize 256) memberi hasil berikut tanpa
latih ulang: resize 256 → 0,9658; 236 → 0,9660; 232 → 0,9638; 224 → 0,9612. Selisih 256 berbanding 236
hanya +0,0002, yaitu satu gambar dan di bawah ambang noise 0,0024.

Sebaliknya, pada bobot pihak lain yang **dilatih** dengan resize 236, mengganti transformasi uji ke
256 menurunkan skornya 0,0055 (0,9754 → 0,9699). Artinya yang berpengaruh adalah kesamaan resolusi
latih dan uji, dan latih-uji yang tidak sama akan menurunkan akurasi tanpa pesan galat. Karena 236
adalah resep resmi, angka itu dipakai di kedua sisi.

## Pemakaian

```bash
python train.py --data data/train --model experiments/convnext_ck224.py \
  --epochs 10 --bs 16 --lr 1e-4 --wd 1e-2 --aug balanced --patience 0 \
  --out runs/putaran2/model.pth --history-out runs/putaran2/history.json
```

`--patience 0` mematikan berhenti dini sehingga `CosineAnnealingLR(T_max=epochs)` melandai sampai
habis. Pada percobaan 4 berhenti dini memutus jadwal saat lr masih 6,5e-5.
