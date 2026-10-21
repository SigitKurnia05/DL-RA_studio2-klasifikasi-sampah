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
| eksperimen 4 | ConvNeXt-Tiny (±28 juta parameter) | 224 | strong | 1e-4 / 1e-2 | 20 → berhenti di 9 | 5 | 0,9658 | 0,9685 | 13 | baseline Putaran 1 |
| **eksperimen 5** | **ConvNeXt-Tiny (±28 juta parameter)** | **224 (resize 236)** | **balanced** | **1e-3 (head 3 ep) / 1e-4 / 5e-2** | **13** | **11** | **0,9753** | **0,9782** | **9** | **submission Putaran 2 (resep dua tahap + bobot kelas)** |

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

## 5. Audit Putaran Preview 2 (21 Oktober 2026)

Putaran Preview 1 menghasilkan macro-F1 **0,9501** dan akurasi **0,9514** pada validation set asisten
(±740 gambar), sementara angka lokal pada split beku 413 gambar adalah 0,9658. Selisih 0,0157 itu
menjadi dasar audit berikut: bagian mana dari konfigurasi yang tidak ikut berpindah ke himpunan
penilaian asisten.

### 5.1 Pembanding eksternal

Tim dengan peringkat 1 pada putaran tersebut memakai backbone yang sama, ConvNeXt-Tiny
(`torchvision`). Bobot submission mereka diukur ulang pada split beku kelompok sendiri memakai
`evaluate.py` resmi, sehingga perbandingannya setara. Bobot tersebut **tidak** dipakai sebagai
submission; yang diambil hanya informasi resep yang memang publik di repositori mereka.

| Skenario | Bobot | Transform | Macro-F1 | Akurasi | Salah |
|---|---|---|---|---:|---:|
| A | kelompok sendiri | resize 256, crop 224 | 0,9658 | 0,9685 | 13 |
| B | peringkat 1 | resize 236, crop 224 | **0,9754** | 0,9782 | 9 |
| C | peringkat 1 | resize 256, crop 224 | 0,9699 | 0,9734 | 11 |
| D | kelompok sendiri + TTA flip horizontal | resize 256, crop 224 | 0,9660 | 0,9685 | 13 |
| E | peringkat 1 + TTA flip horizontal | resize 236, crop 224 | 0,9701 | 0,9734 | 11 |

Tiga kesimpulan yang bisa dibaca dari tabel itu:

1. **Transformasi pengujian saja tidak cukup untuk menjelaskan selisih.** Skenario B berbanding C
   memakai bobot yang identik, dan selisihnya 0,0055 poin hanya karena `resize` 236 berbanding 256.
   Namun uji lanjutan pada § 5.2 menunjukkan bahwa keuntungan itu **tidak gratis** untuk bobot yang
   sudah ada: angka 236 adalah angka yang dipakai model peringkat 1 ketika **dilatih**, sehingga yang
   sebenarnya muncul di sana adalah kesesuaian antara resolusi latih dan resolusi uji, bukan
   transformasi uji yang lebih baik. Resep resmi torchvision untuk bobot ini memang
   `crop_size=224, resize_size=236`, sehingga angka 236 tetap yang dipakai untuk pelatihan putaran
   berikutnya.
2. **TTA flip horizontal tidak dipakai.** Skenario D hanya menambah satu gambar benar pada model
   kelompok sendiri (0,9658 → 0,9660), sedangkan pada model peringkat 1 justru menurunkan skor
   (0,9754 → 0,9701). Rencana TTA pada dokumen versi sebelumnya dibatalkan.
3. **Selisih 0,0094 yang tersisa berasal dari resep pelatihan, bukan dari arsitektur.** Kedua tim
   memakai backbone ConvNeXt-Tiny yang sama persis, namun peringkat 1 memakai pemanasan kepala,
   augmentasi lebih ringan (RandomResizedCrop 0,6–1,0 + flip horizontal + ColorJitter, tanpa flip
   vertikal, RandAugment, maupun RandomErasing), weight decay 0,05, dan 13 epoch total. Karena itu
   perbaikan diarahkan ke resep pelatihan, bukan ke penggantian backbone.

### 5.2 Celah yang ditemukan pada konfigurasi sendiri

Empat nilai `resize` diuji pada bobot submission yang sudah ada (crop 224, tanpa latih ulang) untuk
memisahkan pengaruh transformasi uji dari pengaruh pelatihan:

| `resize` saat uji | Macro-F1 | Akurasi | Salah | F1 `kertas_kardus` |
|---|---:|---:|---:|---:|
| 256 (nilai lama) | 0,9658 | 0,9685 | 13 | 0,9333 |
| **236** (resep resmi) | 0,9660 | 0,9685 | 13 | 0,9333 |
| 232 | 0,9638 | 0,9661 | 14 | 0,9333 |
| 224 (tanpa resize) | 0,9612 | 0,9637 | 15 | 0,9333 |

Membaca tabel ini sekaligus koreksi atas kesimpulan yang sempat tergesa: mengganti transformasi uji
saja **tidak** menaikkan skor secara berarti (+0,0002 = satu gambar, di bawah ambang noise 0,0024).
Angka 236 hanya berarti bila bobotnya juga **dilatih** pada 236, seperti yang dilakukan model
peringkat 1. Aturan yang berlaku: resolusi latih dan resolusi uji harus sama, dan angka 236 adalah
resep resmi torchvision sehingga dipakai untuk pelatihan putaran berikutnya.

| Celah | Bukti | Tindakan |
|---|---|---|
| Resolusi latih tidak mengikuti resep resmi | bobot lama dilatih pada 256 sementara resep resmi ConvNeXt-Tiny `crop_size=224, resize_size=236`; penggantian di sisi uji saja hanya memberi +0,0002 | latih ulang pada 236 |
| Augmentasi `strong` terlalu keras | peringkat 1 memakai augmentasi ringan tetapi menang 0,0094 pada backbone yang sama | mode `balanced` ditambahkan |
| Tanpa pemanasan kepala | peringkat 1 melatih kepala 3 epoch lebih dulu, lalu membuka seluruh layer | resep dua tahap ditambahkan |
| Berhenti dini memotong jadwal | percobaan 4 berhenti di epoch 9 saat lr masih 6,5e-5, sementara epoch 9 masih mencatat 0,9609 | `--patience 0` mematikan berhenti dini |
| Kelas `kertas_kardus` tertinggal | F1 0,9333 (kelompok sendiri) berbanding 0,9434 (peringkat 1); hanya 365 gambar latih | bobot kelas `w_c = N_total / (5·N_c)` |

### 5.3 Rencana putaran Preview 2

Perubahan diterapkan satu variabel sekali ukur, supaya kenaikannya bisa diatribusikan:

1. **crop 224, resize 236**, resep dua tahap (3 epoch kepala lr 1e-3, lalu 10 epoch lr 1e-4, wd 0,05,
   cosine sampai habis), augmentasi `balanced`, bobot kelas penuh. Bidik melewati 0,9754.
2. Bila langkah 1 sudah menembus catatan lama, **crop 320 (resize 448)** dengan resep yang sama.
   93,5% gambar lokal memiliki sisi pendek >= 384 px sehingga resolusi ini masih memakai piksel asli.
3. Yang **tidak** dilakukan pada putaran ini: berpindah ke `timm` (penamaan layer berbeda antar
   versi berisiko menggagalkan `load_state_dict(strict=True)` di environment asisten), rata-rata
   beberapa seed atau SWA (biaya pelatihan tidak sepadan dengan sisa waktu), dan TTA flip.

Seluruh angka pada bagian ini diukur ulang di split beku yang sama dengan `evaluate.py` resmi. Skrip
pengukuran tidak disimpan di repositori karena hanya dipakai sekali untuk audit.

### 5.4 Hasil Eksekusi Putaran 2 (`putaran2_ck224`)

Pelatihan kandidat Putaran 2 berhasil diselesaikan dengan 13 epoch total (3 epoch pemanasan kepala + 10 epoch penyempurnaan seluruh layer). Rekor konvergensi terbaik tercatat pada **epoch 11**:
- **Macro-F1:** `0,9753` (naik +0,0095 dari rekor lama 0,9658).
- **Akurasi:** `0,9782` (naik dari 0,9685).
- **Jumlah Salah:** Berkurang dari 13 menjadi **9 gambar**.
- **Hasil per Kelas:**
  - `botol_plastik`: 1,0000 (78/78 benar, 0 salah).
  - `saset_kemasan`: 0,9854 (101/101 benar, presisi 97,1%).
  - `styrofoam`: 0,9802 (99/101 benar).
  - `gelas_plastik`: 0,9686 (77/78 benar).
  - `kertas_kardus`: 0,9423 (49/55 benar; F1 naik dari 0,9333).
- **Analisis Lanjutan Menuju Macro-F1 >= 0.980:**
  Dari 9 kesalahan yang tersisa, 6 di antaranya berasal dari `kertas_kardus` (3 tertukar ke gelas, 2 ke saset, 1 ke styrofoam). Menariknya, presisi `kertas_kardus` adalah 100% (0 false positives). Hal ini membuktikan model masih terlalu berhati-hati dalam memprediksi kelas kardus. Menaikkan `WEIGHT_POWER` dari 1.0 ke 1.25 pada kandidat final (`experiments/convnext_tiny_final.py`) akan mendorong sampel batas ini terprediksi benar, memangkas 2-3 kesalahan lagi dan menembus target $\ge 0,980$.

