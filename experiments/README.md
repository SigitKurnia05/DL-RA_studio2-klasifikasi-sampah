# Folder `experiments/`

Berisi berkas arsitektur kandidat (berkas `.py` yang mendefinisikan `get_model()` dan
`get_transform()`) yang dipakai saat bereksperimen. Berkas di sini **bukan** submission:
submission resmi tetap di `submission/model.py`.

Alasan pemisahan ini: `train.py --model <berkas>` membangun arsitektur dari berkas yang ditunjuk,
jadi bila eksperimen menunjuk `submission/model.py`, berkas submission ikut berubah setiap kali
backbone dicoba. Dengan folder ini, berkas submission hanya berubah saat memang dipromosikan.

| Berkas | Arsitektur | Resolusi | Status |
|---|---|---|---|
| `convnext_tiny.py` | ConvNeXt-Tiny (torchvision, pretrained ImageNet-1K) | crop 224, resize 256 | kandidat yang dipakai sejak eksperimen 4 |

Kandidat sebelumnya (EfficientNetV2-S, crop 288) didefinisikan langsung di `submission/model.py`
dan masih tersimpan sebagai salinan di `runs/exp3_effnetv2s/model.py`.

Bobot hasil latih **tidak** diletakkan di folder ini — bobot ditulis ke `runs/<tag>/` lewat
argumen `--out` pada `train.py`.

```bash
python train.py --data data/train --model experiments/convnext_tiny.py \
  --out runs/percobaan/model.pth --history-out runs/percobaan/history.json
```
