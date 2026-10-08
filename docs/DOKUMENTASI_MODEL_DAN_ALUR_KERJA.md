# Dokumentasi Arsitektur Pipeline & AI Models Comic Translate
*Panduan Komprehensif untuk Kontributor dan Pengembang (Developer Guide)*

Dokumen ini dirancang sebagai panduan teknis mendalam untuk membantu kontributor dan pengembang memahami, memelihara, dan mengembangkan sistem terjemahan komik **Comic Translate**. Dokumen ini mencakup analisis mendalam terhadap folder `pipeline/` dan `modules/`, spesifikasi input/output, konfigurasi model AI, struktur data inti, serta alur eksekusi dari awal hingga akhir.

---

## Daftar Isi
1. [Arsitektur Umum Sistem](#1-arsitektur-umum-sistem)
2. [Struktur Data Inti: `TextBlock`](#2-struktur-data-inti-textblock)
3. [Analisis Komponen Folder `pipeline/`](#3-analisis-komponen-folder-pipeline)
   - [3.1 Main Orchestrator (`main_pipeline.py`)](#31-main-orchestrator-main_pipelinepy)
   - [3.2 Block Detection Handler (`block_detection.py`)](#32-block-detection-handler-block_detectionpy)
   - [3.3 OCR Handler (`ocr_handler.py`)](#33-ocr-handler-ocr_handlerpy)
   - [3.4 Translation Handler (`translation_handler.py`)](#34-translation-handler-translation_handlerpy)
   - [3.5 Inpainting Handler (`inpainting.py` & `inpainting_boxes.py`)](#35-inpainting-handler-inpaintingpy--inpainting_boxespy)
   - [3.6 Batch & Webtoon Processor (`batch_processor.py` & `webtoon_batch/`)](#36-batch--webtoon-processor)
   - [3.7 Cache Manager (`cache_manager.py`)](#37-cache-manager-cache_managerpy)
   - [3.8 Segmentation Handler (`segmentation_handler.py`)](#38-segmentation-handler-segmentation_handlerpy)
4. [Analisis Modul AI Folder `modules/`](#4-analisis-modul-ai-folder-modules)
   - [4.1 Modul Deteksi (`modules/detection/`)](#41-modul-deteksi-modulesdetection)
   - [4.2 Modul OCR (`modules/ocr/`)](#42-modul-ocr-modulesocr)
   - [4.3 Modul Inpainting (`modules/inpainting/`)](#43-modul-inpainting-modulesinpainting)
   - [4.4 Modul Penerjemahan (`modules/translation/`)](#44-modul-penerjemahan-modulestranslation)
   - [4.5 Modul Rendering (`modules/rendering/`)](#45-modul-rendering-modulesrendering)
5. [Contoh Kode API untuk Pengembang (Python API Usage)](#5-contoh-kode-api-untuk-pengembang-python-api-usage)

---

## 1. Arsitektur Umum Sistem

Comic Translate menggunakan pendekatan **Modular Pipeline Architecture**. Setiap tahapan pengolahan komik dipisahkan menjadi modul tersendiri (*Detection*, *OCR*, *Translation*, *Inpainting*, *Rendering*) yang dikoordinasikan oleh pengelola alur kerja utama (`ComicTranslatePipeline`).

```
                    ┌─────────────────────────────────────────┐
                    │               Input Image               │
                    │   (Single Image / Webtoon Stream / CBZ) │
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          ComicTranslatePipeline                                 │
│                                                                                 │
│  ┌─────────────────────────┐     ┌─────────────────────────┐                    │
│  │ BlockDetectionHandler   │ ──► │      OCRHandler         │                    │
│  │ (RT-DETR-v2 Engine)     │     │ (PPOCR / MangaOCR / LLM)│                    │
│  └────────────┬────────────┘     └────────────┬────────────┘                    │
│               │                               │                                 │
│               ▼                               ▼                                 │
│  ┌─────────────────────────┐     ┌─────────────────────────┐                    │
│  │    InpaintingHandler    │     │   TranslationHandler    │                    │
│  │  (LaMa / AOT-GAN Engine)│     │  (GPT / Claude / Gemini)│                    │
│  └────────────┬────────────┘     └────────────┬────────────┘                    │
│               │                               │                                 │
│               └────────────────┬──────────────┘                                 │
│                                │                                                │
│                                ▼                                                │
│                  ┌──────────────────────────┐                                   │
│                  │  Text Wrapping & Render  │                                   │
│                  │  (PySide / PIL Render)   │                                   │
│                  └──────────────────────────┘                                   │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │          Final Rendered Output          │
                    └─────────────────────────────────────────┘
```

---

## 2. Struktur Data Inti: `TextBlock`

Seluruh informasi mengenai teks, posisi, orientasi, serta hasil terjemahan pada sebuah halaman komik disimpan dalam objek `TextBlock` (`modules/utils/textblock.py`). Objek ini mengalir di sepanjang pipeline dari modul deteksi hingga modul rendering.

### Atribut Inti `TextBlock`
| Atribut | Tipe Data | Deskripsi |
| :--- | :--- | :--- |
| `xyxy` | `np.ndarray` | Bounding box teks `[x1, y1, x2, y2]` dalam piksel. |
| `bubble_xyxy` | `np.ndarray` | Bounding box balon kata pengurung (jika ada). |
| `text_class` | `str` | Klasifikasi blok: `'text_bubble'` atau `'text_free'`. |
| `lines` | `List` | Koordinat garis-garis teks individu di dalam blok. |
| `text` | `str` | Teks asli hasil pengenalan OCR. |
| `translation` | `str` | Teks terjemahan hasil keluaran modul Translation. |
| `direction` | `str` | Arah baca: `'horizontal'` atau `'vertical'`. |
| `source_lang` | `str` | Kode/nama bahasa sumber (misal: `'Japanese'`, `'Korean'`). |
| `target_lang` | `str` | Kode/nama bahasa target (misal: `'English'`, `'Indonesian'`). |
| `alignment` | `str` | Penjajaran teks: `'center'`, `'left'`, `'right'`. |
| `font_color` | `str \| tuple` | Warna teks (misal: `"#000000"` atau RGB tuple). |
| `min_font_size` / `max_font_size` | `int` | Batas ukuran font saat di-render. |

### Metode Inti `TextBlock`
* `xywh`: Property mengembalikan `[x1, y1, width, height]`.
* `center`: Property mengembalikan titik tengah `[x_center, y_center]`.
* `source_lang_direction`: Mengembalikan `'ver_rtl'` jika arah vertikal, atau `'hor_ltr'`.
* `deep_copy()`: Membuat salinan mendalam (*deep copy*) dari `TextBlock`.
* `sort_blk_list(blk_list, right_to_left)`: Mengurutkan daftar `TextBlock` sesuai alur baca alami komik (atas ke bawah, kanan ke kiri untuk manga vertikal).

---

## 3. Analisis Komponen Folder `pipeline/`

Folder `pipeline/` berisi komponen pengatur alur bisnis (*business logic orchestrators*) yang menghubungkan antarmuka antarmuka pengguna (GUI/Viewer) dengan modul AI di `modules/`.

---

### 3.1 Main Orchestrator (`main_pipeline.py`)
Kelas `ComicTranslatePipeline` memegang referensi ke seluruh handler dan manager:
* **Inisialisasi**: Menghubungkan `CacheManager`, `BlockDetectionHandler`, `InpaintingHandler`, `OCRHandler`, `TranslationHandler`, `SegmentationHandler`, `BatchProcessor`, dan `WebtoonBatchProcessor`.
* **Pembersihan Memori (`release_model_caches`)**: Membebaskan memori RAM/VRAM dengan mengosongkan cache ONNX Session dan PyTorch engine, serta memicu Python Garbage Collector (`gc.collect()`).

---

### 3.2 Block Detection Handler (`block_detection.py`)
Bertanggung jawab atas alur deteksi blok teks pada halaman:
* **Alur Eksekusi**:
  1. Mengambil gambar aktif dari `ImageViewer`.
  2. Memanggil `TextBlockDetector.detect(image)`.
  3. Mengoptimalkan batas render balon kata via `get_best_render_area`.
  4. Mengurutkan blok sesuai urutan baca (`sort_blk_list`).
  5. Mengirimkan signal UI (`blk_detail_signal`, `render_display_signal`) untuk memperbarui kanvas.

---

### 3.3 OCR Handler (`ocr_handler.py`)
Pengelola eksekusi OCR untuk halaman penuh, area terlihat (webtoon), atau blok tunggal:
* **Alur Caching**: Sebelum mengeksekusi OCR, handler meminta `CacheManager._get_ocr_cache_key`. Jika hasil OCR sudah ada di cache, data diambil langsung tanpa menjalankan model OCR.
* **Integrasi Factory**: Menggunakan `OCRFactory.create_engine(settings, source_lang, ocr_model, backend)` untuk memperoleh instance OCR yang sesuai.
* **Mode Area Terlihat (Webtoon)**: Menggunakan `filter_and_convert_visible_blocks` untuk menjalankan OCR hanya pada blok teks yang sedang berada di dalam viewport pengguna.

---

### 3.4 Translation Handler (`translation_handler.py`)
Mengelola alur penerjemahan teks menggunakan Large Language Models (LLM):
* **Context Handling**: Mengirimkan seluruh teks halaman beserta gambar (opsional) dan `extra_context` ke modul `Translator`.
* **Single Block vs Full Page**:
  * Untuk *Single Block*, mengecek apakah terjemahan sudah ada di cache.
  * Untuk *Full Page*, menerjemahkan seluruh blok teks dalam satu prompt untuk menjaga konsistensi konteks naratif.
* **Transformasi**: Mengaplikasikan format huruf kapital (*uppercase*) pada `blk.translation` jika dipilih oleh pengguna.

---

### 3.5 Inpainting Handler (`inpainting.py` & `inpainting_boxes.py`)
Pengelola pembersihan teks dari gambar:
* **Pembuatan Mask (`inpainting_boxes.py`)**:
  * Menggabungkan bounding box balon kata dan garis teks (`ppocr_lines` / `heuristic_lines`).
  * Menerapkan ekspansi/dilasi mask agar piksel tepi teks terhapus bersih.
* **Eksekusi Inpainting**:
  * Menjalankan model inpainting (`LaMa` / `AOT-GAN`).
  * `get_inpainted_patches`: Memotong patch hasil inpainting berdasarkan area mask untuk di-blend kembali ke gambar original, menghemat konsumsi memori.

---

### 3.6 Batch & Webtoon Processor (`batch_processor.py` & `webtoon_batch/`)
* **BatchProcessor (`batch_processor.py`)**:
  Memproses sekumpulan gambar secara sekuensial atau paralel melalui alur: `Detect -> OCR -> Translate -> Inpaint -> Render -> Save`.
* **WebtoonBatchProcessor (`pipeline/webtoon_batch/`)**:
  Memproses gambar webtoon yang sangat panjang dengan teknik **Seam-Aware Virtual Page Streaming**:
  * `virtual_page.py`: Menggabungkan beberapa file webtoon menjadi kanvas virtual yang kontinyu.
  * `chunk.py`: Memotong kanvas virtual menjadi chunk-chunk kecil di area kosong (*seam*) agar tidak memotong balon kata.
  * `flow.py`: Mengontrol eksekusi pipeline per-chunk.
  * `render.py`: Menggabungkan kembali chunk yang telah di-inpaint dan di-render menjadi file gambar output tunggal.

---

### 3.7 Cache Manager (`cache_manager.py`)
Mengelola caching hasil pemrosesan agar tidak terjadi komputasi/panggilan API berulang:
* **Dua Level Cache**:
  1. *In-Memory Cache*: Menyimpan hasil berbasis dictionary selama aplikasi berjalan.
  2. *Disk Cache*: Menyimpan snapshot cache di direktori lokal.
* **Generasi Key**:
  * **OCR Key**: Hash SHA256 dari `(image_crop_bytes, source_lang, ocr_model)`.
  * **Translation Key**: Hash SHA256 dari `(full_page_text, source_lang, target_lang, translator_model, extra_context)`.
  * **Inpainting Key**: Hash SHA256 dari `(image_bytes, mask_bytes, inpainter_model)`.

---

### 3.8 Segmentation Handler (`segmentation_handler.py`)
Menangani pemotongan halaman webtoon secara horizontal berdasarkan analisis kecerahan piksel/baris kosong untuk memudahkan rendering dan pengolahan per-panel.

---

## 4. Analisis Modul AI Folder `modules/`

Folder `modules/` berisi implementasi teknis model AI, pengolah gambar, serta pustaka pendukung.

---

### 4.1 Modul Deteksi (`modules/detection/`)

#### Model Utama: RT-DETR-v2
* **Arsitektur**: Real-Time Detection Transformer v2 (`ogkalu/comic-text-and-bubble-detector`).
* **Kelas Keluaran**: `text_bubble` (label 0), `text_free` (label 1).

#### Spesifikasi Input & Output
* **Input**:
  * `img`: NumPy array RGB, bentuk `[H, W, 3]`, tipe `uint8`.
* **Output**:
  * `List[TextBlock]`: Daftar objek `TextBlock` dengan koordinat `xyxy` terisi.
* **Konfigurasi Engine (`DetectionEngineFactory`)**:
  * `backend`: `'onnx'` (via `RTDetrV2ONNXDetection`) atau `'torch'` (via `RTDetrV2Detection`).
  * `device`: Auto-resolved (`'cuda'`, `'cpu'`, `'directml'`, `'mps'`).

#### Pustaka Pendukung Line Detection
* `ppocr_lines.py`: Menggunakan PP-OCR detection head untuk mendeteksi garis teks di dalam bounding box.
* `heuristic_lines/`: Menggunakan analisis kontur biner dan proyeksi horizontal/vertikal untuk memisahkan garis teks tanpa GPU.

---

### 4.2 Modul OCR (`modules/ocr/`)

`OCRFactory` bertugas membuat engine OCR berdasarkan bahasa sumber dan model yang dipilih.

#### Engine Lokal vs Cloud/LLM
| Name / Key | Class / Engine | Target Bahasa / Script | Mode Eksekusi |
| :--- | :--- | :--- | :--- |
| **MangaOCR** | `MangaOCRMobileONNXEngine` / `MangaOCREngine` | Japanese (Kanji, Kana, Vertical) | Local (ONNX / Torch) |
| **PororoOCR** | `PororoOCREngineONNX` / `PororoOCREngine` | Korean (Hangul) | Local (ONNX / Torch) |
| **PPOCRv5** | `PPOCRv5Engine` / `PPOCRv5TorchEngine` | Chinese, Cyrillic, Latin, English | Local (ONNX / Torch) |
| **GPT-4.1-mini** | `GPTOCR` | Semua Bahasa | Cloud API (OpenAI) |
| **Gemini-2.5** | `GeminiOCR` | Semua Bahasa | Cloud API (Google) |
| **Microsoft Azure** | `MicrosoftOCR` | Semua Bahasa | Cloud API (Azure Vision) |
| **Google Cloud** | `GoogleOCR` | Semua Bahasa | Cloud API (Google Vision) |

#### Spesifikasi Input & Output Engine OCR
* **Input**:
  * `img`: NumPy array RGB dari crop area teks `[H_crop, W_crop, 3]`.
* **Output**:
  * `str`: Teks hasil pengenalan (*recognized string*).

---

### 4.3 Modul Inpainting (`modules/inpainting/`)

Modul Inpainting merekonstruksi piksel gambar di bawah mask teks.

#### Model-Model Inpainting
1. **LaMa (`lama.py`)**:
   * Model utama berbasis *Fast Fourier Convolutions* (FFCs).
   * Finetuned pada dataset Anime/Manga (`dreMaz/AnimeMangaInpainting`).
   * Sangat stabil pada resolusi bervariasi karena sifat Fourier transform.
2. **AOT-GAN (`aot.py`)**:
   * *Aggregated Contextual Transformations GAN*.
   * Menggunakan receptive fields bertingkat untuk mengisi tekstur kompleks.
3. **MI-GAN (`mi_gan.py`)**:
   * *Modulated Invertible GAN* untuk kasus inpainting tertentu.

#### Spesifikasi Input & Output Inpainting
* **Input**:
  * `image`: NumPy array RGB, bentuk `[H, W, 3]`, tipe `uint8`.
  * `mask`: NumPy array 2D, bentuk `[H, W]`, tipe `uint8` (`0` = background, `255` = area yang di-inpaint).
  * `config`: Objek `Config` (konfigurasi parameter tambahan).
* **Preprocessing & Normalisasi Tensor**:
  * **LaMa**: Gambar diubah ke float32 `[0, 1]` via `norm_img(image)`.
  * **AOT-GAN**: Gambar di-scale ke range `[-1.0, 1.0]` (`(image / 127.5) - 1.0`).
* **Output**:
  * `np.ndarray`: Gambar RGB ter-inpaint, bentuk `[H, W, 3]`, tipe `uint8`.

---

### 4.4 Modul Penerjemahan (`modules/translation/`)

Menggunakan `TranslationFactory` dan `Translator` (`modules/translation/processor.py`).

#### Alur Eksekusi Penerjemahan
1. Mengumpulkan daftar `TextBlock` pada halaman.
2. Menyusun prompt JSON / Structured Text yang berisi seluruh ID blok dan teks aslinya.
3. Mengirimkan prompt ke LLM (GPT-4.1, Claude-4.5, Gemini-2.5, DeepSeek, atau OpenRouter API).
4. Memuaskan terjemahan kembali ke properti `blk.translation` untuk masing-masing ID `TextBlock`.

---

### 4.5 Modul Rendering (`modules/rendering/`)

Menangani penataan letak (*word wrapping*) dan penggambaran teks terjemahan ke atas gambar.

#### Komponen Utama
* **`pyside_word_wrap`**:
  * Menggunakan **Pencarian Biner (Binary Search)** untuk menentukan ukuran font terbesar (antara `min_font_size` dan `max_font_size`) agar teks muat di dalam bounding box (`roi_width`, `roi_height`).
* **`VerticalTextDocumentLayout`**:
  * Layout khusus PySide6 Qt Text Document untuk mendukung teks vertikal CJK (Jepang/Korea/Mandarin) dengan alur baca dari kanan ke kiri.
* **`hyphen_textwrap.py`**:
  * Algoritma wrapping kata greedy berbasis pemotongan tanda hubung (*hyphenation*) untuk bahasa berbasis Latin.
* **`draw_text`**:
  * Menggunakan PIL `ImageDraw.multiline_text` atau PySide Qt Scene untuk menggambar teks terjemahan beserta efek stroke/outline putih/hitam di sekitar teks.

---

## 5. Contoh Kode API untuk Pengembang (Python API Usage)

Berikut adalah contoh lengkap penggunaan modul-modul secara mandiri tanpa menggunakan GUI PySide6.

```python
import cv2
import numpy as np

# 1. Loading Gambar Input
image_bgr = cv2.imread("sample_manga_page.jpg")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# ---------------------------------------------------------
# 2. DETEKSI TEKS & BALON KATA (Detection)
# ---------------------------------------------------------
from modules.detection.factory import DetectionEngineFactory

# Inisialisasi Detection Engine (RT-DETR-v2 via ONNX)
class MockSettings:
    def is_gpu_enabled(self): return True
    def get_tool_selection(self, tool): return 'RT-DETR-v2'

settings = MockSettings()
detector = DetectionEngineFactory.create_engine(settings, model_name='RT-DETR-v2', backend='onnx')

# Jalankan deteksi
text_blocks = detector.detect(image_rgb)
print(f"[Detection] Ditemukan {len(text_blocks)} blok teks.")

# ---------------------------------------------------------
# 3. OPTICAL CHARACTER RECOGNITION (OCR)
# ---------------------------------------------------------
from modules.ocr.factory import OCRFactory

# Inisialisasi OCR Engine untuk Bahasa Jepang (MangaOCR)
ocr_engine = OCRFactory.create_engine(
    settings=settings,
    source_lang_english='Japanese',
    ocr_model='Default',
    backend='onnx'
)

# Jalankan OCR pada tiap crop blok teks
for i, blk in enumerate(text_blocks):
    x1, y1, x2, y2 = map(int, blk.xyxy)
    crop = image_rgb[y1:y2, x1:x2]

    if crop.size > 0:
        recognized_text = ocr_engine.recognize(crop)
        blk.text = recognized_text
        print(f"  [OCR Block {i}] Teks Asli: {recognized_text}")

# ---------------------------------------------------------
# 4. INPAINTING (Pembersihan Teks)
# ---------------------------------------------------------
from modules.inpainting.lama import LaMa
from modules.inpainting.schema import Config

# Membuat Mask Biner berdasarkan bounding box
h, w, _ = image_rgb.shape
mask = np.zeros((h, w), dtype=np.uint8)
for blk in text_blocks:
    x1, y1, x2, y2 = map(int, blk.xyxy)
    mask[y1:y2, x1:x2] = 255

# Inisialisasi dan jalankan Model LaMa Inpainting
inpainter = LaMa()
inpainter.init_model(device="cuda", backend="onnx")
cleaned_image_rgb = inpainter.forward(image_rgb, mask, Config())
print("[Inpainting] Pembersihan latar belakang selesai.")

# ---------------------------------------------------------
# 5. TRANSLATION & RENDERING
# ---------------------------------------------------------
from modules.rendering.render import draw_text

# Contoh mengisi terjemahan secara manual (atau via Translator API)
for blk in text_blocks:
    blk.translation = "Hello World!"  # Hasil terjemahan

# Draw Teks ke atas gambar yang bersih
final_rendered_rgb = draw_text(
    image=cleaned_image_rgb,
    blk_list=text_blocks,
    font_pth="fonts/comic_font.ttf",
    colour="#000000",
    init_font_size=36,
    min_font_size=12,
    outline=True
)

# Save Hasil Akhir
final_rendered_bgr = cv2.cvtColor(final_rendered_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite("translated_output.jpg", final_rendered_bgr)
print("[Export] Gambar hasil terjemahan berhasil disimpan ke 'translated_output.jpg'.")
```

---

## 6. Ringkasan Modul AI & Parameter Konfigurasi

| Modul | Class / Factory | Input Format | Output Format | Opsi Backend / Parameter Kunci |
| :--- | :--- | :--- | :--- | :--- |
| **Detection** | `DetectionEngineFactory` | `np.ndarray` RGB `[H, W, 3]` | `List[TextBlock]` | `backend='onnx'\|'torch'`, `device`, `shrink_bbox` |
| **OCR** | `OCRFactory` | `np.ndarray` Crop RGB `[H, W, 3]` | `str` (Recognized text) | `source_lang_english`, `ocr_model`, API credentials |
| **Inpainting** | `LaMa` / `AOT` / `MI-GAN` | Image RGB `[H, W, 3]`, Mask `[H, W]` | `np.ndarray` Inpainted RGB `[H, W, 3]` | JIT `.pt` vs ONNX `.onnx`, `pad_mod=8`, auto-cast |
| **Translation** | `Translator` | `List[TextBlock]`, Context String | `blk.translation` string | `source_lang`, `target_lang`, LLM model selection |
| **Rendering** | `draw_text` / `pyside_word_wrap` | Clean Image RGB, `List[TextBlock]` | Rendered `np.ndarray` RGB | `font_pth`, `init_font_size`, `outline`, `alignment` |
