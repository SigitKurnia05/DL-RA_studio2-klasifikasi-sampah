"""Kandidat arsitektur: EfficientNetV2-S pretrained ImageNet-1K, crop 288 px.

Berkas ini adalah **kandidat eksperimen**, bukan submission. Isinya identik dengan
`submission/model.py` pada kurun eksperimen 2 dan 3, dan disimpan di sini agar berkas
submission tidak perlu diubah setiap kali backbone dicoba.

Kontrak (sama dengan yang diminta `evaluate.py`):
  - `get_model()` hanya membangun arsitektur; tidak mengunduh bobot dan tidak memuat
    `model.pth` (yang memuat adalah `evaluate.py`, dengan `strict=True`).
  - `get_transform()` adalah transform validasi saat pelatihan. Nilainya harus sama persis
    dengan yang dipakai `train.py`; bila berbeda, akurasi turun tanpa pesan galat.
  - Keluaran forward: logits [batch, 5] dengan urutan kelas alfabetis menurut `evaluate.py`
    (0 botol_plastik, 1 gelas_plastik, 2 kertas_kardus, 3 saset_kemasan, 4 styrofoam).

Catatan hasil: kandidat ini mencapai macro-F1 0,9590 (eksperimen 2, Colab) dan 0,9595
(eksperimen 3, lokal) pada split validasi beku 413 gambar. Bila disimpan dalam fp32,
berkas bobotnya berukuran sekitar 78 MB sehingga masih di bawah batas 100 MB GitHub.

Catatan resolusi: dilatih pada RandomResizedCrop 288 px dan dinilai pada CenterCrop 288
dari sisi pendek 320 px. Kedua angka ini hanya didefinisikan di berkas ini, agar `train.py`
dan `get_transform()` tidak dapat berbeda tanpa terlihat.
"""
import torch.nn as nn
from torchvision import models, transforms

NUM_CLASSES = 5
CROP = 288  # sisi crop saat latih maupun saat uji
RESIZE = 320  # sisi pendek sebelum center-crop uji

# Penunjuk enum bobot ImageNet untuk `train.py`. Ini bukan pengunduhan: pengambilan bobot
# dilakukan `train.py`, sedangkan `get_model()` tetap membangun arsitektur kosong.
IMAGENET_WEIGHTS = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1


def get_model():
    """Bangun EfficientNetV2-S dengan kepala 5 kelas (bobot acak, tanpa unduhan).

    Returns:
        torch.nn.Module: EfficientNetV2-S yang mengeluarkan logits [batch, 5].
    """
    model = models.efficientnet_v2_s(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
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
