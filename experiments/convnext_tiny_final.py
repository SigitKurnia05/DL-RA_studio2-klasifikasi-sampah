"""Kandidat Final: ConvNeXt-Tiny crop 224, resize 236 (target macro-F1 >= 0.980).

Evolusi dari `experiments/convnext_ck224.py`:
  - Mempertahankan rasio resmi torchvision (crop 224, resize 236) yang terbukti
    memangkas error dari 13 menjadi 9 gambar.
  - Meningkatkan WEIGHT_POWER dari 1.0 ke 1.25. Analisis matriks konfusi menunjukkan
    bahwa presisi kelas `kertas_kardus` sudah 100% (0 false positives), namun recall-nya
    masih 89.1% (6 false negatives, tertukar ke gelas_plastik dan saset_kemasan).
    Penaikan bobot 1.25x memberikan dorongan penalti yang cukup agar sampel batas
    terklasifikasi benar ke kertas_kardus tanpa memicu false positive.
  - Skema pelatihan 2-tahap: 3 epoch pemanasan kepala klasifikasi (lr 1e-3),
    dilanjutkan 14 epoch fine-tuning seluruh layer (lr 1e-4, wd 0.05, CosineAnnealingLR).

Kontrak submission tetap terjaga:
  - get_model() membangun arsitektur kosong tanpa unduhan.
  - get_transform() transform deterministik evaluasi.
  - Keluaran forward: logits [batch, 5] berurutan alfabetis.
"""
import torch.nn as nn
from torchvision import models, transforms

NUM_CLASSES = 5
CROP = 224
RESIZE = 236

IMAGENET_WEIGHTS = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1

# ── Resep Pelatihan Final (dibaca oleh train.py) ──────────────────────────────
HEAD_EPOCHS = 3
HEAD_LR = 1e-3
CLASS_WEIGHTED = True
WEIGHT_POWER = 1.25  # dorong recall kertas_kardus untuk mencapai F1 >= 0.980


def get_model():
    """Bangun ConvNeXt-Tiny dengan kepala 5 kelas (bobot acak, tanpa unduhan)."""
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, NUM_CLASSES)
    return model


def get_transform():
    """Transform validasi deterministik (resep resmi ConvNeXt-Tiny)."""
    return transforms.Compose([
        transforms.Resize(RESIZE),
        transforms.CenterCrop(CROP),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
