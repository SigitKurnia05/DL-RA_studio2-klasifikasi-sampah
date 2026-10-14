"""Model candidate: ConvNeXt-Tiny pretrained ImageNet-1K, crop 224 px.

Arsitektur:
  - ConvNeXt-Tiny (torchvision.models.convnext_tiny)
  - Parameter: ~28M
  - Resolusi input: crop 224x224 (resize 256x256)
  - Classifier head: Linear(768 -> 5)

Kontrak model (sesuai evaluasi Studio 2):
  - get_model() membangun arsitektur kosong tanpa download bobot.
  - get_transform() transform deterministik validasi 224x224.
  - Output forward: logits [batch, 5] (urutan alfabetis CLASSES).
"""
import torch.nn as nn
from torchvision import models, transforms

NUM_CLASSES = 5
CROP = 224
RESIZE = 256

IMAGENET_WEIGHTS = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1


def get_model():
    """Bangun ConvNeXt-Tiny dengan kepala 5 kelas (bobot acak, tanpa unduhan)."""
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, NUM_CLASSES)
    return model


def get_transform():
    """Transform validasi: PIL (RGB) -> Tensor [3, 224, 224]."""
    return transforms.Compose([
        transforms.Resize(RESIZE),
        transforms.CenterCrop(CROP),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
