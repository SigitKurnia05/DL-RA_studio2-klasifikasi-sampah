# Analisis dan Perbandingan Percobaan — Studio 2 (Klasifikasi Sampah)

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027
**Kelompok:** 11 (YOLO) · Putaran Preview 1 (14 Oktober 2026)

Data: 2.753 gambar latih pada 5 kelas; split validasi beku 413 gambar (seed 42, hash kanonik
`36bee95ec932553bd3bf89d8eabe8c1d562d1343f65a442235c1df403be7a423`).
Metrik penilaian: macro-F1. Fungsi kerugian: `CrossEntropyLoss(label_smoothing=0,1)` dengan
optimizer AdamW dan CosineAnnealingLR.

Seluruh angka pada dokumen ini diukur dengan `evaluate.py` resmi pada split beku yang sama, dan
berkas buktinya tersedia di `runs/<tag>/`. Ringkasan interaktif beserta grafiknya terdapat pada
`notebooks/riwayat_eksperimen.ipynb`.

## 1. Ringkasan Angka

| Percobaan | Arsitektur | Crop | Augmentasi | lr / wd | Epoch | Epoch terbaik | Macro-F1 | Akurasi | Salah | Catatan |
|---|---|---|---|---|---|---|---|---|---|---|
| eksperimen 1 | EfficientNetV2-S (±21,5 juta parameter) | 288 | standar | 3e-4 / 1e-4 | 12 | 3 | 0,9555 | 0,9564 | 18 | val loss melonjak mulai epoch 6 (overfitting) |
| eksperimen 2 | EfficientNetV2-S | 288 | strong | 1,5e-4 / 1e-2 | 12 | 9 | 0,9590 | 0,9613 | 16 | val loss stabil |
| eksperimen 3 | EfficientNetV2-S | 288 | strong | 1,5e-4 / 1e-2 | 11 | 7 | 0,9595 | 0,9613 | 16 | setara eksperimen 2; selisih berupa noise |
| eksperimen 4 | ConvNeXt-Tiny (±28 juta parameter) | 224 | strong | 1e-4 / 1e-2 | 20 → berhenti di 9 | 5 | **0,9658** | 0,9685 | 13 | submission saat ini |

Baseline asisten: ResNet-18 pretrained ≈ 0,92 pada validation set tersembunyi (±740 gambar).
Angka tersebut berasal dari himpunan yang berbeda sehingga **tidak dapat dibandingkan 1:1**;
fungsinya hanya sebagai patokan.

## 2. Justifikasi Hasil

### 2.1 Eksperimen 1 → 2: pengendalian overfitting (0,9555 → 0,9590)

Pada eksperimen 1, model berukuran 21,5 juta parameter dilatih pada 2.340 gambar latih dengan
augmentasi standar dan weight decay lemah (1e-4). Gejala overfitting tampak jelas pada kurva:
`train_loss` terus menurun, tetapi `val_loss` melonjak drastis pada epoch 6 (0,50 → 1,42). Model
mulai menghafal data latih dan kehilangan kemampuan generalisasi, sehingga macro-F1 terbaiknya
justru tercatat pada epoch 3 (sebelum lonjakan) lalu menurun.

Eksperimen 2 memperbaikinya melalui tiga penyesuaian:

1. **Augmentasi kuat** (RandomResizedCrop 0,5–1,0, RandAugment(2, 9), RandomErasing, dan flip
   vertikal) memperluas distribusi data latih secara sintetis. Model dipaksa mempelajari invarian
   (rotasi, pemotongan, oklusi, warna) alih-alih menghafal piksel tertentu.
2. **Weight decay dinaikkan menjadi 1e-2** (regularisasi L2) untuk menahan magnitudo bobot agar
   tidak tumbuh berlebihan.
3. **Learning rate diturunkan menjadi 1,5e-4**, karena lr yang terlalu besar turut memicu lonjakan
   val loss pada eksperimen 1.

Hasilnya `val_loss` bertahan pada kisaran 0,49–0,50 sepanjang 12 epoch tanpa lonjakan, macro-F1 naik
menjadi 0,9590, dan jumlah gambar salah turun dari 18 menjadi 16.

### 2.2 Eksperimen 2 berbanding 3: selisih 0,9590 berbanding 0,9595 adalah noise

Kedua percobaan memakai konfigurasi yang sama persis; perbedaannya hanya tempat menjalankannya
(Colab berbanding lokal) dan urutan augmentasi acak yang berbeda. Selisih 0,0005 berada jauh di bawah
ambang ±0,01, sedangkan pada split 413 gambar satu gambar bernilai ±0,0024 poin macro-F1.
Kesimpulannya, eksperimen 2 dan 3 merepresentasikan model yang sama dan tidak ada alasan teknis
untuk memilih salah satunya.

### 2.3 Eksperimen 2/3 → 4: penggantian backbone ke ConvNeXt-Tiny (0,9590 → 0,9658)

Kenaikan sebesar 0,0068 terhadap eksperimen 2 (0,0063 terhadap eksperimen 3, yang merupakan rekor
sebelumnya) disertai penurunan jumlah gambar salah dari 16 menjadi 13. Yang perlu dicatat,
ConvNeXt-Tiny menang **meskipun** diuji pada crop yang lebih kecil (224 berbanding 288).
Penjelasannya:

- **Kualitas backbone lebih menentukan daripada resolusi input** pada transfer learning.
  ConvNeXt-Tiny (±28 juta parameter) merupakan "ResNet yang dimodernisasi" melalui depthwise
  convolution, inverted bottleneck, LayerNorm, dan GELU, dan secara empiris menyaingi maupun
  mengungguli ViT serta EfficientNet pada ImageNet-1K. EfficientNetV2-S (±21,5 juta parameter)
  dioptimalkan untuk efisiensi FLOPs melalui NAS yang sadar biaya pelatihan, bukan untuk akurasi
  mentah tertinggi.
- Representasi fitur ConvNeXt-Tiny yang lebih kaya memberi kepala klasifikasi masukan yang lebih
  diskriminatif, sehingga lima kelas sampah yang sebagian mirip secara visual lebih mudah dipisahkan.
- Eksperimen 4 juga memakai learning rate lebih rendah (1e-4) yang lebih stabil pada backbone besar.

Bukti per kelas: ConvNeXt-Tiny menaikkan F1 `botol_plastik` (0,9747 → 0,9935), `saset_kemasan`
(0,9600 → 0,9854), dan `kertas_kardus` (0,9273 → 0,9333), sedangkan `gelas_plastik` turun tipis
(0,9677 → 0,9565).

### 2.4 Peran early stopping dan checkpoint berdasarkan macro-F1

Eksperimen 4 menjadi bukti langsungnya: bobot terbaik jatuh pada **epoch 5** (macro-F1 0,9658),
setelah itu `val_loss` kembali naik (0,4663 → 0,4837 → 0,4957) dan tidak pernah memecahkan rekor.
Early stopping aktif pada epoch 9. Bila yang diambil adalah bobot epoch terakhir dan bukan checkpoint
terbaik, skornya justru lebih rendah. Mekanisme "simpan bobot hanya saat macro-F1 membaik" pada
`train.py` inilah yang mencegah model terlanjur memburuk.

### 2.5 Penyebab `kertas_kardus` selalu menjadi kelas terlemah

Terdapat dua penyebab, dan keduanya terlihat pada data:

1. **Ketidakseimbangan kelas.** `kertas_kardus` hanya berisi 365 gambar, paling sedikit
   (saset_kemasan 673, styrofoam 671, botol_plastik 523, gelas_plastik 521), sehingga paling jarang
   dilihat model.
2. **Kemiripan visual.** Kertas dan kardus berwarna terang bertekstur mudah tertukar dengan styrofoam
   dan gelas plastik. Pada confusion matrix eksperimen 4, dari 55 gambar `kertas_kardus` terdapat
   5 gambar yang meleset (3 ke gelas_plastik, 1 ke saset_kemasan, 2 ke styrofoam), sementara
   `styrofoam` juga meleset 5 kali (2 ke gelas_plastik, 1 ke kertas_kardus, 2 ke saset_kemasan).
   Kedua kelas ini adalah pasangan yang paling sering tertukar.

Hal ini pula yang menjadikan macro-F1 sebagai metrik penilaian, bukan akurasi. Akurasi akan condong
ke kelas mayoritas (saset_kemasan dan styrofoam), sedangkan macro-F1 memberi setiap kelas bobot yang
sama sehingga kelas kecil seperti `kertas_kardus` tetap dihitung setara.

## 3. Kesimpulan dan Rekomendasi

- Model yang dipakai saat ini: **ConvNeXt-Tiny crop 224, augmentasi strong, lr 1e-4, wd 1e-2**,
  dengan macro-F1 0,9658 pada split validasi beku. Angka ini melampaui baseline asisten (0,92),
  dengan catatan bahwa himpunan penilaiannya berbeda sehingga perbandingan tersebut bersifat
  indikatif.
- Jalur perbaikan yang sudah terbukti: (1) meredam overfitting melalui augmentasi kuat dan weight
  decay, (2) mengganti backbone ke arsitektur yang lebih kuat, dan (3) mengandalkan early stopping
  beserta pemilihan checkpoint berdasarkan macro-F1.

Langkah berikutnya untuk putaran Preview 2, diurutkan dari yang paling murah:

1. Menaikkan resolusi ConvNeXt-Tiny ke crop 320 atau 384; pada crop 224 backbone ini masih
   menyisakan ruang.
2. Memperbaiki kelas lemah `kertas_kardus` melalui oversampling atau class-weighted loss, dan bila
   diizinkan menambah data.
3. Test-time augmentation berupa flip horizontal saat inferensi; menambah F1 secara tipis dengan
   waktu evaluasi dua kali lipat, masih jauh di bawah batas 15 menit.
4. Mencoba backbone setingkat (ConvNeXt-Small) atau arsitektur modern lain (Swin/Swin-v2) bila
   langkah 1 dan 2 sudah tidak memberi kenaikan.

## 4. Catatan Teknis pada Komit Berikutnya

- `submission/model.py` berisi ConvNeXt-Tiny crop 224, konsisten dengan `submission/model.pth`
  sehingga aman dimuat oleh `evaluate.py` dengan `strict=True`.
- Berkas bobot berukuran **53,1 MB dalam fp16**, bukan 106,2 MB dalam fp32. Konversi dilakukan karena
  GitHub menolak berkas di atas 100 MB. Skor pada split beku tidak berubah (0,9658 pada fp32 maupun
  fp16, 0 prediksi berbeda dari 413 gambar).
- Berkas arsitektur kandidat berada di `experiments/`, terpisah dari `submission/model.py`, agar
  berkas submission hanya berubah pada saat promosi.
- Riwayat dan kurva tiap percobaan di `runs/<tag>/` ikut di-commit sebagai bukti, sedangkan bobot di
  dalamnya tidak (lihat `.gitignore`).
