"""Kandidat A: ConvNeXt-Tiny crop 224, resize 236 (resep resmi torchvision).

Bedanya dengan `experiments/convnext_tiny.py`:
  - RESIZE 236, bukan 256. Angka ini yang tertulis pada meta bobot ImageNet-1K
    torchvision untuk ConvNeXt-Tiny (`crop_size=224, resize_size=236`), sehingga
    konteks tepi gambar tidak terpotong lebih banyak dari yang dilihat backbone
    saat dilatih di ImageNet.
  - Menyertakan resep pelatihan (HEAD_EPOCHS, HEAD_LR, CLASS_WEIGHTED, WEIGHT_POWER)
    yang dibaca `train.py`. Nilai resep sengaja ditaruh di berkas arsitektur supaya
    satu percobaan = satu berkas, dan tidak ada arsitektur yang tersilang resep.

Kontrak submission tetap sama:
  - `get_model()` hanya membangun arsitektur; tidak mengunduh bobot dan tidak memuat
    `model.pth` (yang memuat `evaluate.py`, dengan `strict=True`).
  - `get_transform()` adalah transform validasi saat pelatihan; angka CROP/RESIZE
    hanya boleh ada di berkas ini agar latih dan uji tidak dapat berbeda diam-diam.
  - Keluaran forward: logits [batch, 5] dengan urutan kelas alfabetis menurut
    `evaluate.py` (0 botol_plastik, 1 gelas_plastik, 2 kertas_kardus,
    3 saset_kemasan, 4 styrofoam).
"""
import torch.nn as nn
from torchvision import models, transforms

NUM_CLASSES = 5
CROP = 224  # sisi crop saat latih maupun saat uji
RESIZE = 236  # sisi pendek sebelum center-crop uji (resep resmi ConvNeXt-Tiny)

# Penunjuk enum bobot ImageNet untuk `train.py`. Ini bukan pengunduhan: pengambilan
# bobot dilakukan `train.py`, sedangkan `get_model()` tetap membangun arsitektur kosong.
IMAGENET_WEIGHTS = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1

# ── Resep pelatihan (dibaca train.py) ─────────────────────────────────────────
# Tahap 1: kepala klasifikasi saja, dengan backbone dibekukan.
HEAD_EPOCHS = 3
HEAD_LR = 1e-3
# Pembobotan kelas dari frekuensi data latih: w_c = (N_total / (5 * N_c)) ** WEIGHT_POWER.
# WEIGHT_POWER 1.0 membobot kelas langka secara penuh (rasio 1.84x karena
# kertas_kardus hanya 365 gambar); 0.5 memakai akar rasio bila 1.0 terlalu agresif;
# 0.0 mematikan pembobotan. Angka ini diubah di sini, bukan di train.py.
CLASS_WEIGHTED = True
WEIGHT_POWER = 1.0


def get_model():
    """Bangun ConvNeXt-Tiny dengan kepala 5 kelas (bobot acak, tanpa unduhan).

    Returns:
        torch.nn.Module: ConvNeXt-Tiny yang mengeluarkan logits [batch, 5].
    """
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, NUM_CLASSES)
    return model


def get_transform():
    """Transform validasi: PIL (RGB) -> Tensor [3, CROP, CROP].

    Returns:
        torchvision.transforms.Compose: transform deterministik untuk penilaian.
    """
    return transforms.Compose([
        transforms.Resize(RESIZE),
        transforms.CenterCrop(CROP),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
