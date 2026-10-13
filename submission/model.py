"""Submission Studio 2 — EfficientNetV2-S pretrained ImageNet, crop 288 px.

Kontrak submission (sama dengan yang dijalankan evaluate.py):
  - get_model() hanya MEMBANGUN arsitektur; tidak mengunduh bobot, tidak memuat
    model.pth (evaluate.py yang memuatnya dengan strict=True).
  - get_transform() = transform validasi saat pelatihan. Harus sama persis dengan
    yang dipakai train.py, kalau tidak, akurasi jatuh tanpa terlihat.
  - Output forward: logits [batch, 5], urutan kelas alfabetis milik evaluate.py
    (0 botol_plastik, 1 gelas_plastik, 2 kertas_kardus, 3 saset_kemasan, 4 styrofoam).

Alasan memilih EfficientNetV2-S (bukan convnext_nano/ResNet):
  - Berasal dari torchvision, jadi tidak menambah paket dan bentuk arsitekturnya
    stabil antar-versi library. Ini penting karena evaluate.py memuat bobot dengan
    strict=True: satu nama layer yang bergeser saja membuat submission gagal muat.
  - 21,5 juta parameter -> berkas bobot sekitar 86 MB, masih di bawah batas 100 MB
    GitHub sehingga cukup di-commit langsung tanpa Release/LFS (batas teknis 500 MB).

Catatan resolusi: dilatih pada RandomResizedCrop 288 px, dinilai pada CenterCrop 288
dari sisi pendek 320 px. Angka-angka ini hanya ada di sini, supaya train.py dan
get_transform() tidak bisa berbeda diam-diam.
"""
import torch.nn as nn
from torchvision import models, transforms

NUM_CLASSES = 5
CROP = 288  # sisi crop saat latih maupun saat uji
RESIZE = 320  # sisi pendek sebelum center-crop uji

# Sumber bobot ImageNet untuk train.py. Ini hanya penunjuk enum, bukan pengunduhan:
# pengambilan bobotnya dilakukan train.py, sedangkan get_model() tetap membangun
# arsitektur kosong sesuai kontrak submission.
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
