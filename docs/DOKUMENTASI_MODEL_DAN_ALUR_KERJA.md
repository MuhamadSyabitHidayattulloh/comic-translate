# Dokumentasi Model dan Alur Kerja Comic Translate

Dokumen ini berisi panduan penggunaan model-model AI yang digunakan dalam **Comic Translate** (Deteksi Teks & Balon Kata, OCR, dan Inpainting), penjelasan alur kerja (*end-to-end workflow*) lengkap, serta **Panduan API untuk Developer (Kode Python)**.

---

## 1. Panduan dan Cara Penggunaan Models

Comic Translate memanfaatkan pendekatan modular untuk memproses gambar komik/manga/webtoon. Setiap modul dihubungkan melalui *factory pattern* yang fleksibel, mendukung eksekusi berbasis **ONNX Runtime** maupun **PyTorch**, dengan akselerasi GPU (CUDA/DirectML/ROCm/MPS) atau CPU.

---

### 1.1 Deteksi Teks dan Balon Kata (Text & Bubble Detection)

Proses deteksi bertujuan menemukan lokasi balon kata (*speech bubble*) serta garis/blok teks bebas pada halaman komik.

#### Model yang Digunakan
* **RT-DETR-v2** (`ogkalu/comic-text-and-bubble-detector`): Model *real-time detection transformer* yang telah dilatih secara khusus pada lebih dari 11.000 gambar komik (Manga, Webtoon, dan Komik Barat).

#### Backend & Eksekusi
* **PyTorch (`RTDetrV2Detection`)**: Digunakan saat PyTorch tersedia dan backend diatur ke Torch.
* **ONNX Runtime (`RTDetrV2ONNXDetection`)**: Digunakan secara default untuk efisiensi memori dan kompatibilitas lintas platform.
* Terintegrasi melalui `DetectionEngineFactory`.

#### Kategori / Kelas Deteksi
1. `text_bubble`: Balon kata yang mengurung teks.
2. `text_free` / `text_block`: Teks bebas atau efek suara di luar balon kata.

#### Cara Penggunaan & Konfigurasi
* **Melalui GUI Settings**: Pada panel pengaturan, pilih *Detector* = **RT-DETR-v2**.
* **Threshold & Area Rendering**:
  * Hasil deteksi akan menghasilkan daftar objek `TextBlock` yang berisi koordinat `xyxy` (box) dan `xywh`.
  * Fungsi `get_best_render_area` dan `shrink_bbox` melakukan penyesuaian otomatis batas render berdasarkan bentuk balon kata agar teks terjemahan pas di tengah balon kata.

---

### 1.2 OCR (Optical Character Recognition)

Modul OCR bertugas membaca teks asli dari potongan area yang telah terdeteksi.

#### Arsitektur & Model yang Didukung
Comic Translate mendukung mesin OCR lokal (tanpa internet) dan mesin OCR berbasis Cloud / LLM melalui `OCRFactory`.

1. **Engine OCR Lokal (Default Off-line)**:
   * **Jepang (Japanese)**: `MangaOCR` (`manga_ocr` berbasis PyTorch) atau `MangaOCRMobileONNXEngine` (ONNX). Sangat akurat membaca huruf Kanji, Hiragana, Katakana, dan teks vertikal.
   * **Korea (Korean)**: `PororoOCR` (`pororo` PyTorch / `PororoOCREngineONNX`).
   * **Mandarin, Cyrillic (Rusia), dan Latin (Inggris, Prancis, Jerman, Spanyol, Belanda, Italia)**: `PPOCRv5` (`PPOCRv5Engine` ONNX / `PPOCRv5TorchEngine`).

2. **Engine OCR berbasis Cloud & LLM**:
   * **GPT-4.1-mini** (`GPTOCR`): Mengirimkan potongan area/halaman ke OpenAI GPT vision API.
   * **Gemini-2.5-Flash-Lite** (`GeminiOCR`): Menggunakan Google Gemini Vision API.
   * **Microsoft Azure Vision** (`MicrosoftOCR`): Menggunakan Azure Cognitive Services.
   * **Google Cloud Vision** (`GoogleOCR`): Menggunakan Google Cloud Vision API.
   * **UserOCR**: Untuk pengguna berakun yang terhubung dengan server pengolahan remote.

#### Cara Penggunaan & Konfigurasi
* Pilih **Source Language** (Bahasa Sumber) pada antarmuka utama (misal: *Japanese*, *Korean*, *Chinese*, *English*, dll.).
* Pilih **OCR Engine** pada panel Settings (Default, GPT, Gemini, Microsoft Azure, atau Google Cloud).
* Pengolahan OCR dijalankan secara otomatis per-blok (`TextBlock.text`) atau per-halaman penuh jika menggunakan model LLM Vision.

---

### 1.3 Inpainting (Penghapusan Teks & Pembersihan Gambar)

Modul Inpainting berfungsi menghapus teks asli dari gambar dan merekonstruksi latar belakang komik agar bersih sebelum teks terjemahan digambar.

#### Model yang Digunakan
* **LaMa (Large Mask Inpainting)**: Model inpainting berbasis *Fast Fourier Convolutions* yang di-finetune khusus untuk komik/anime (`dreMaz/AnimeMangaInpainting`). Tersedia dalam format PyTorch JIT (`.pt`) dan ONNX (`.onnx`).
* **AOT-GAN (Aggregated Contextual Transformations)**: Model inpainting berbasis GAN yang handal mempertahankan tekstur pada resolusi tinggi.
* **MI-GAN (Modulated Invertible GAN)**: Alternatif pembersihan area tertentu.

#### Pembuatan Mask (Text Masking)
* **Otomatik (Automatic Mode)**: Mask dibuat secara otomatis dari bounding box teks/balon kata, ditambah dengan ekstraksi garis teks (`ppocr_lines` atau `heuristic_lines`) dan pembesaran mask (*dilation*) untuk memastikan seluruh piksel teks tertutup sempurna.
* **Manual Mode**: Pengguna dapat menggunakan alat kuas (*brush tool*) pada kanvas GUI untuk menggambar mask secara manual di area yang sulit dibersihkan otomatis.

#### Cara Penggunaan
* Pengaturan model inpainting berada di Settings -> Inpainter (misal: **LaMa** atau **AOT-GAN**).
* Proses inpainting menghasilkan *patch* gambar yang bersih, yang kemudian ditimpa kembali pada kanvas utama (`get_inpainted_patches`).

---

## 2. Alur Kerja Lengkap (End-to-End Workflow)

Berikut adalah tahapan alur kerja lengkap dari saat gambar diunggah hingga hasil rendering terjemahan ditampilkan/disimpan:

```
[1. Upload Image]
       │
       ▼
[2. Text & Bubble Detection (RT-DETR-v2)] ──► Menghasilkan TextBlock List
       │
       ▼
[3. OCR (MangaOCR / Pororo / PPOCR / LLM)] ──► Teks Asli (source text)
       │
       ▼
[4. Translation (GPT / Claude / Gemini)] ──► Teks Terjemahan (translation)
       │
       ▼
[5. Inpainting (LaMa / AOT-GAN)] ──► Latar Belakang Bersih (inpainted image)
       │
       ▼
[6. Text Wrapping & Rendering] ──► Menyesuaikan Ukuran Font, Wrap, Vertical/Horizontal
       │
       ▼
[7. Final Render & Export] ──► Tampilan GUI Canvas / Simpan Gambar
```

---

### Detail Langkah demi Langkah

#### Tahap 1: Pengunggahan & Memuat Gambar (*Upload & Image Loading*)
1. Pengguna memuat satu gambar atau banyak gambar (batch) melalui GUI, drag-and-drop, atau memilih berkas/arsip (JPG, PNG, WEBP, CBZ, CBR).
2. Sistem membaca gambar ke dalam memori sebagai array NumPy RGB dan menampilkannya di `ImageViewer` / Kanvas Qt.
3. Untuk mode Webtoon (`WebtoonBatchProcessor`), halaman-halaman panjang dipotong secara virtual (*virtual page streaming*) untuk memproses area yang terlihat.

#### Tahap 2: Deteksi Teks & Balon Kata (*Text & Bubble Detection*)
1. Gambar dikirim ke `BlockDetectionHandler` -> `TextBlockDetector`.
2. Model `RT-DETR-v2` melakukan deteksi dan mengembalikan daftar bounding box.
3. Bounding box diklasifikasikan menjadi balon kata (`text_bubble`) atau teks bebas (`text_free`).
4. Algoritma `get_best_render_area` menentukan batas area penulisan terjemahan terbaik di dalam balon kata.
5. Koordinat disimpan dalam daftar objek `TextBlock`.

#### Tahap 3: Pengenalan Teks (*OCR - Optical Character Recognition*)
1. Gambar dipotong (*crop*) sesuai area tiap `TextBlock`.
2. `OCRHandler` memanggil engine OCR dari `OCRFactory` berdasarkan **Source Language** dan model OCR yang dipilih.
3. Teks hasil pembacaan OCR dimasukkan ke dalam properti `blk.text` pada masing-masing `TextBlock`.

#### Tahap 4: Penerjemahan Teks (*Translation*)
1. `TranslationHandler` mengambil teks dari seluruh `TextBlock` pada halaman tersebut beserta konteks ekstra (*extra context*).
2. Teks dikirim ke modul penerjemah (`Translator`) berbasis LLM (seperti GPT-4.1, Claude-4.5, Gemini-2.5, atau DeepSeek).
3. Hasil terjemahan disimpan pada properti `blk.translation`.
4. Jika opsi kapitalisasi (*uppercase*) diaktifkan, teks terjemahan dikonversi menjadi huruf kapital.
5. Sistem caching (`CacheManager`) menyimpan hasil penerjemahan untuk menghindari pemanggilan API berulang yang tidak perlu.

#### Tahap 5: Inpainting / Pembersihan Latar Belakang (*Inpainting*)
1. `InpaintingHandler` membuat mask hitam-putih di area teks/balon kata yang akan dibersihkan.
2. Gambar dan mask dikirim ke model Inpainting (misal: `LaMa`).
3. Model merekonstruksi piksel gambar yang tertutup teks sehingga menghasilkan gambar latar belakang yang bersih tanpa teks asli.
4. Patch yang telah dibersihkan disisipkan kembali ke gambar utama (`get_inpainted_patches`).

#### Tahap 6: Wrap Teks & Rendering Akhir (*Text Wrapping & Rendering*)
1. `manual_wrap` / `pyside_word_wrap` menerima teks terjemahan, jenis font, warna, outline, line spacing, dan batas lebar/tinggi blok.
2. **Pencarian Biner Ukuran Font**: Algoritma menghitung ukuran font terbesar (antara `min_font_size` dan `max_font_size`) yang membuat seluruh teks terjemahan muat di dalam kotak/balon kata.
3. **Pemesanan Tata Letak (Horizontal vs Vertikal)**:
   * Untuk bahasa CJK (Jepang/Mandarin/Korea) dengan arah vertikal, digunakan `VerticalTextDocumentLayout`.
   * Untuk bahasa dengan spasi (seperti Inggris/Indonesia/Spanyol), digunakan algoritma wrapping kata greedy dengan penanganan tanda hubung (*hyphen wrap*).
4. **Drawing & Compositing**: Teks digambar di atas gambar/patch yang di-inpaint menggunakan PySide6 Qt Graphics Scene atau PIL `ImageDraw` dengan efek stroke/outline jika diaktifkan.

#### Tahap 7: Tampilan Akhir & Ekspor (*Final Display & Export*)
1. Gambar terjemahan akhir ditampilkan secara *real-time* pada viewer aplikasi Comic Translate.
2. Pengguna dapat melakukan penyesuaian manual (misal: mengedit teks terjemahan, mengubah posisi box, mengganti font, atau melakukan manual inpainting).
3. Gambar terjemahan disimpan ke disk (format PNG/JPG/WEBP atau dikemas kembali ke CBZ/CBR).

---

## 3. Panduan Developer / Pemrograman (Python API Usage)

Berikut adalah contoh penggunaan langsung modul-modul AI dalam kode Python untuk pengembang yang ingin mengintegrasikan modul secara terpisah atau memperluas fungsionalitas.

### 3.1 Contoh Deteksi Teks & Balon Kata (Detection)

```python
import cv2
from modules.detection.factory import DetectionEngineFactory

# Load gambar (Format RGB NumPy array)
image_bgr = cv2.imread("sample_comic.jpg")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Inisialisasi engine deteksi via Factory (model default RT-DETR-v2)
detector_engine = DetectionEngineFactory.create_engine(
    settings=settings_obj,  # Objek konfigurasi settings
    model_name='RT-DETR-v2',
    backend='onnx'  # 'onnx' atau 'torch'
)

# Menjalankan deteksi -> mengembalikan list of TextBlock
text_blocks = detector_engine.detect(image_rgb)

for blk in text_blocks:
    print(f"Jenis Blok: {blk.text_class}")  # 'text_bubble' atau 'text_free'
    print(f"Koordinat Box [x1, y1, x2, y2]: {blk.xyxy}")
```

### 3.2 Contoh Optical Character Recognition (OCR)

```python
from modules.ocr.factory import OCRFactory

# Membuat engine OCR untuk Bahasa Jepang
ocr_engine = OCRFactory.create_engine(
    settings=settings_obj,
    source_lang_english='Japanese',
    ocr_model='Default',  # MangaOCR untuk Jepang, PPOCRv5 untuk Latin/Chinese, dll.
    backend='onnx'
)

# Pemotongan area teks dari gambar berdasarkan TextBlock
x1, y1, x2, y2 = text_blocks[0].xyxy
crop_img = image_rgb[int(y1):int(y2), int(x1):int(x2)]

# Eksekusi OCR
recognized_text = ocr_engine.recognize(crop_img)
text_blocks[0].text = recognized_text
print("Teks Terdeteksi:", recognized_text)
```

### 3.3 Contoh Inpainting (Pembersihan Latar Belakang)

```python
import numpy as np
from modules.inpainting.lama import LaMa
from modules.inpainting.schema import Config

# Inisialisasi Model Inpainting LaMa
inpainter = LaMa()
inpainter.init_model(device="cuda", backend="onnx")

# Membentuk Mask Hitam-Putih (2D NumPy array: 255 untuk teks/mask, 0 untuk background)
h, w, _ = image_rgb.shape
mask = np.zeros((h, w), dtype=np.uint8)

for blk in text_blocks:
    x1, y1, x2, y2 = map(int, blk.xyxy)
    mask[y1:y2, x1:x2] = 255

# Eksekusi Inpainting
cleaned_image_rgb = inpainter.forward(image_rgb, mask, Config())

# cleaned_image_rgb adalah gambar RGB uint8 yang bebas teks
```

### 3.4 Contoh Penerjemahan Teks (Translation)

```python
from modules.translation.processor import Translator

# Inisialisasi Penerjemah (misal: Jepang -> Indonesia / Inggris)
translator = Translator(
    main_page=main_page_obj,
    source_lang="Japanese",
    target_lang="English"
)

# Memproses daftar TextBlock secara batch beserta konteks halaman
translator.translate(text_blocks, image_rgb, extra_context="Comic translation context")

for blk in text_blocks:
    print(f"Asli: {blk.text} | Terjemahan: {blk.translation}")
```

### 3.5 Contoh Rendering Teks (Text Rendering)

```python
from modules.rendering.render import draw_text

# Menggambar teks terjemahan di atas gambar yang telah di-inpaint
rendered_image = draw_text(
    image=cleaned_image_rgb,
    blk_list=text_blocks,
    font_pth="fonts/anime_font.ttf",
    colour="#000000",
    init_font_size=40,
    min_font_size=10,
    outline=True
)

# Simpan gambar hasil akhir
rendered_bgr = cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR)
cv2.imwrite("translated_comic.jpg", rendered_bgr)
```

---

## 4. Ringkasan Penggunaan Ringkas

| Modul | Model Utama | Output Utama | Opsi Utama / Parameter |
| :--- | :--- | :--- | :--- |
| **Detection** | RT-DETR-v2 | Daftar Bounding Box (`TextBlock`) | ONNX / PyTorch backend, BBox Shrink |
| **OCR** | MangaOCR / Pororo / PPOCRv5 / Cloud LLM | Teks Asli (`blk.text`) | Bahasa Sumber, Mode Per-Blok / Per-Halaman |
| **Translation**| GPT-4.1 / Claude-4.5 / Gemini-2.5 | Teks Terjemahan (`blk.translation`) | Bahasa Target, Extra Context, Uppercase |
| **Inpainting** | LaMa / AOT-GAN / MI-GAN | Gambar Latar Bersih | Mask Auto / Manual, Dilasi Mask |
| **Rendering** | PySide6 Qt / PIL / Hyphen Wrap | Canvas & Image Render | Font Family, Font Size Auto-fit, Vertical Layout, Stroke/Outline |
