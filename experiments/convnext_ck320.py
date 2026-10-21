"""Kandidat B: ConvNeXt-Tiny crop 320, resize 448 (resolusi dinaikkan).

Hanya dipakai BILA kandidat A (`experiments/convnext_ck224.py`) sudah mengalahkan
catatan 0,9754 pada split beku. Aturan mainnya: satu variabel sekali ubah, jadi
resolusi dinaikkan setelah resep dasarnya terbukti.

Kenapa 448 dan bukan 384: ImageNet-1K menyediakan evaluasi resmi pada 384 (resize 448)
untuk ConvNeXt-Tiny, artinya bobot pretrained sudah pernah dipakai pada rentang
resolusi itu. Dataset lokal pun mendukung: 93,5% gambar memiliki sisi pendek >= 384 px,
jadi crop 320 tidak kehabisan piksel asli.

Catatan risiko yang harus diukur, bukan diasumsikan: 448 px adalah 4x piksel 224 px,
sehingga waktu latih per epoch naik sekitar 4x dan kebutuhan VRAM naik. Bila VRAM
Colab tidak cukup, turunkan `--bs` (batch 8) sebelum menurunkan resolusi.

Resep pelatihan, kontrak submission, dan urutan kelas sama dengan kandidat A.
"""
import torch.nn as nn
from torchvision import models, transforms

NUM_CLASSES = 5
CROP = 320  # sisi crop saat latih maupun saat uji
RESIZE = 448  # sisi pendek sebelum center-crop uji

IMAGENET_WEIGHTS = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1

# ── Resep pelatihan (dibaca train.py) ─────────────────────────────────────────
HEAD_EPOCHS = 3
HEAD_LR = 1e-3
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
