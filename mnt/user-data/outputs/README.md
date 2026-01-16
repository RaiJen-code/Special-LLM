# 🚀 Panduan Lengkap: Eksperimen LLM + RAG vs LLM Standalone pada Jetson Orin Nano

## 📋 Daftar Isi
1. [Overview](#overview)
2. [Arsitektur Sistem](#arsitektur-sistem)
3. [File-File yang Tersedia](#file-file-yang-tersedia)
4. [Persiapan Awal](#persiapan-awal)
5. [Langkah-Langkah Eksekusi](#langkah-langkah-eksekusi)
6. [Struktur Hasil](#struktur-hasil)
7. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

Proyek ini bertujuan untuk membandingkan performa antara:
- **LLM + RAG** (Retrieval-Augmented Generation) - LLM dengan akses ke knowledge base
- **LLM Standalone** - LLM tanpa RAG

### Platform
- **Hardware**: NVIDIA Jetson Orin Nano 8GB
- **Software**: JetPack 6.2.1, CUDA 12.6.77, TensorRT 10.7.0.23, cuDNN 9.17.1.4
- **Models**: LLaMA 3.2 3B (via Ollama), Whisper tiny (ASR), SentenceTransformer (Embeddings), Piper (TTS)

### Metrik yang Diukur
1. **Performa**:
   - Response Latency (total time)
   - LLM Inference Time
   - Component Breakdown (ASR, RAG, LLM, TTS)

2. **Kualitas** (jika ada reference answers):
   - ROUGE-1 & ROUGE-L scores
   - BLEU score
   - Semantic Similarity

3. **Resource Usage**:
   - Memory consumption
   - Processing efficiency

---

## 🏗️ Arsitektur Sistem

```
┌─────────────┐
│   Audio     │
│   Input     │
└──────┬──────┘
       │
       v
┌─────────────┐     ┌──────────────┐
│   Whisper   │────>│   Enhanced   │
│   (ASR)     │     │   Assistant  │
└─────────────┘     │              │
                    │   ┌────────┐ │
                    │   │  RAG   │ │
                    │   │ System │ │
                    │   └────────┘ │
                    │       │      │
                    │       v      │
                    │   ┌────────┐ │
                    │   │  LLM   │ │
                    │   │(Ollama)│ │
                    │   └────────┘ │
                    └───────┬──────┘
                            │
                            v
                    ┌──────────────┐
                    │   Piper TTS  │
                    └──────┬───────┘
                           │
                           v
                    ┌──────────────┐
                    │ Audio Output │
                    └──────────────┘
```

---

## 📁 File-File yang Tersedia

### 1. **enhanced_assistant.py**
   - Main module yang sudah disempurnakan
   - Fitur:
     * Enhanced RAG dengan PDF processing
     * Vector database (FAISS) dengan chunking strategy
     * Performance tracking dan logging
     * Flexible configuration
     * Support untuk testing mode (tanpa audio I/O)

### 2. **01_Preparation_and_Setup.ipynb**
   - Notebook untuk persiapan dan verifikasi sistem
   - Fungsi:
     * Load dan organize test questions
     * Prepare reference answers
     * Verify system components
     * Check all dependencies

### 3. **experiment_notebook.ipynb** (akan dibuat)
   - Notebook utama untuk running experiments
   - Fungsi:
     * Run comprehensive testing (RAG vs Non-RAG)
     * Collect all metrics
     * Statistical analysis
     * Generate visualizations

---

## 🔧 Persiapan Awal

### Langkah 1: Transfer Files ke Jetson

```bash
# Di Jetson Orin Nano, buat direktori project
mkdir -p ~/voice_assistant_research
cd ~/voice_assistant_research

# Copy file enhanced_assistant.py
# Copy file 01_Preparation_and_Setup.ipynb
# Copy file referensi.pdf
# Copy file jurnal_tambahan.pdf
# Copy file List_Pertanyaan.docx
```

### Langkah 2: Install Dependencies (Sekali Saja dengan Internet)

```bash
# Install Python dependencies
pip install jupyter notebook pandas matplotlib seaborn \
            scikit-learn python-docx PyPDF2 faiss-cpu \
            sentence-transformers --break-system-packages

# Verify Whisper is installed (already done in tutorial)
python3 -c "import whisper; print('Whisper OK')"

# Download SentenceTransformer model (sekali saja)
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Verify Ollama and model
ollama list | grep llama3.2

# Verify Piper TTS
ls -la /home/rangga/piper/build/piper
ls -la /usr/local/share/piper/models/en_US-lessac-medium.onnx
```

### Langkah 3: Verifikasi Struktur Direktori

Setelah semua file ditransfer, struktur direktori harus seperti ini:

```
~/voice_assistant_research/
├── enhanced_assistant.py
├── 01_Preparation_and_Setup.ipynb
├── experiment_notebook.ipynb (akan dibuat)
├── referensi.pdf
├── jurnal_tambahan.pdf
├── List_Pertanyaan.docx
├── assets/
│   ├── bip.wav
│   └── bip2.wav
├── data/ (akan dibuat otomatis)
├── results/ (akan dibuat otomatis)
└── logs/ (akan dibuat otomatis)
```

---

## 🚀 Langkah-Langkah Eksekusi

### FASE 1: Persiapan (OFFLINE - Tidak perlu internet)

#### Step 1: Start Ollama Server

```bash
# Di terminal pertama, jalankan Ollama server
ollama serve
```

> **Penting**: Biarkan terminal ini tetap berjalan selama eksperimen!

#### Step 2: Start Jupyter Notebook

```bash
# Di terminal kedua
cd ~/voice_assistant_research
jupyter notebook --ip=0.0.0.0 --no-browser
```

Buka browser dan akses:
```
http://<jetson-ip>:8888
```

#### Step 3: Run Preparation Notebook

1. Buka **01_Preparation_and_Setup.ipynb**
2. Run semua cells secara berurutan (Cell → Run All)
3. Verifikasi bahwa:
   - ✅ All libraries loaded
   - ✅ All questions loaded (20 questions total)
   - ✅ Reference answers prepared
   - ✅ Ollama server running
   - ✅ Whisper model loaded
   - ✅ Embedding model loaded
   - ✅ Piper TTS ready
   - ✅ PDF documents found

> **Catatan**: Jika ada error, lihat bagian [Troubleshooting](#troubleshooting)

---

### FASE 2: Testing Mode Sederhana (OFFLINE)

Sebelum melakukan eksperimen lengkap, kita test dulu dengan beberapa pertanyaan.

#### Quick Test (Terminal Mode)

```bash
cd ~/voice_assistant_research

# Test dengan 3 pertanyaan sample
python3 enhanced_assistant.py test
```

Ini akan:
- Test sistem dengan 3 pertanyaan
- Generate responses dengan RAG
- Save results ke `logs/test_results_[timestamp].json`
- Memverifikasi bahwa semua komponen bekerja

**Expected Output:**
```
==================================================
VOICE ASSISTANT - Text Testing Mode
==================================================
✓ Ollama server is running
✓ Available models: X

[Query 1/3] What is the main function of a resistor?
Response: [LLM response here]
Time: X.XXs

[Query 2/3] Explain how PWM works
Response: [LLM response here]
Time: X.XXs

[Query 3/3] Is capacitor used to store charge? True or false?
Response: [LLM response here]
Time: X.XXs

Test results saved to: logs/test_results_[timestamp].json
```

---

### FASE 3: Eksperimen Lengkap (OFFLINE)

Sekarang kita siap untuk eksperimen yang komprehensif!

#### Step 1: Open Main Experiment Notebook

Buka **experiment_notebook.ipynb** yang sudah saya buat sebelumnya di Jupyter.

#### Step 2: Configure Experiment

Di cell konfigurasi, Anda bisa mengatur:

```python
# Configuration untuk eksperimen
NUM_REPETITIONS = 3  # Jumlah repetisi per pertanyaan
USE_FULL_DATASET = True  # True = semua pertanyaan, False = subset untuk quick test
```

**Estimasi Waktu:**
- Full dataset (20 questions × 3 reps × 2 modes) = 120 tests
- Estimasi: ~3-5 detik per test = **6-10 menit total**

#### Step 3: Run Experiment

Jalankan cells secara berurutan. Notebook akan:

1. **Setup** (Cells 1-4):
   - Import libraries
   - Load questions dan reference answers
   - Setup evaluation metrics

2. **Testing** (Cell 5-6):
   - Run all questions dengan mode **Non-RAG** (LLM standalone)
   - Run all questions dengan mode **RAG** (LLM + RAG)
   - Setiap pertanyaan ditest 3 kali
   - Progress ditampilkan real-time

**Progress akan terlihat seperti ini:**
```
================================================================================
STARTING MAIN EXPERIMENT
================================================================================

[1/20] Testing: What is the main function of a resistor...
Category: simple
  [1/120] Non-RAG - Rep 1/3... ✓ (2.34s)
  [2/120] RAG - Rep 1/3... ✓ (2.56s)
  [3/120] Non-RAG - Rep 2/3... ✓ (2.41s)
  [4/120] RAG - Rep 2/3... ✓ (2.48s)
  [5/120] Non-RAG - Rep 3/3... ✓ (2.39s)
  [6/120] RAG - Rep 3/3... ✓ (2.52s)
  Progress: 5.0% | Elapsed: 0.2min | ETA: 9.5min

[2/20] Testing: Name three examples of sensors...
...
```

3. **Save Results** (Cell 7):
   - Raw results → `experiment_results/raw_results_[timestamp].json`
   - CSV format → `experiment_results/results_[timestamp].csv`

4. **Statistical Analysis** (Cells 8-9):
   - Performance summary (RAG vs Non-RAG)
   - Performance by category
   - RAG retrieval statistics

5. **Visualizations** (Cells 10-14):
   - Response time comparison (box plots)
   - Time by category (grouped bar chart)
   - Time distribution (histograms)
   - RAG analysis (documents retrieved)
   - Summary comparison

6. **Generate Report** (Cell 15):
   - Comprehensive text report
   - Statistical summary
   - Comparative analysis

7. **Export for Journal** (Cells 16-17):
   - Summary table (ready for paper)
   - Category-wise comparison

---

### FASE 4: Analisis Hasil (OFFLINE)

Setelah eksperimen selesai, semua file hasil akan tersimpan di:

```
~/voice_assistant_research/experiment_results/
├── raw_results_[timestamp].json
├── results_[timestamp].csv
├── experiment_report_[timestamp].txt
├── journal_summary_[timestamp].csv
├── category_comparison_[timestamp].csv
├── conclusions_[timestamp].txt
└── plots/
    ├── response_time_comparison.png
    ├── time_by_category.png
    ├── time_distribution.png
    ├── rag_analysis.png
    └── summary_comparison.png
```

#### Melihat Hasil

1. **Baca Report:**
```bash
cat experiment_results/experiment_report_[timestamp].txt
```

2. **Buka Plots:**
Plots sudah dalam format PNG dan bisa langsung dilihat atau dibuka di browser.

3. **Analisis Data Lebih Lanjut:**
File CSV bisa dibuka di Excel atau LibreOffice untuk analisis tambahan.

---

## 📊 Struktur Hasil

### 1. Raw Results JSON
Berisi data mentah semua test runs:
```json
[
  {
    "question": "What is the main function of a resistor?",
    "use_rag": false,
    "mode": "Non-RAG",
    "category": "simple",
    "repetition": 1,
    "llm_time": 2.341,
    "total_time": 2.345,
    "response": "A resistor is...",
    "success": true,
    "timestamp": "2025-01-16T10:23:45"
  },
  ...
]
```

### 2. CSV Results
Format tabel untuk analisis:
```
question_id,category,mode,repetition,llm_time,total_time,response,success,...
simple_1,simple,Non-RAG,1,2.341,2.345,"A resistor...",true,...
simple_1,simple,RAG,1,2.567,2.571,"A resistor...",true,...
...
```

### 3. Experiment Report
Text report dengan summary lengkap:
```
================================================================================
COMPREHENSIVE EXPERIMENT REPORT
================================================================================

1. EXPERIMENTAL SETUP
----------------------------------------
Total Questions Tested: 20
Repetitions per Question: 3
Total Tests Conducted: 120
Successful Tests: 120 (100.0%)

2. PERFORMANCE METRICS SUMMARY
----------------------------------------
NON-RAG (LLM Standalone):
  LLM Inference Time:
    - Mean: 2.345s
    - Std Dev: 0.123s
    ...

RAG (LLM + RAG):
  LLM Inference Time:
    - Mean: 2.567s
    - Std Dev: 0.145s
    ...

3. COMPARATIVE ANALYSIS
----------------------------------------
LLM Inference Time:
  RAG vs Non-RAG: +9.47% slower

Total Processing Time:
  RAG vs Non-RAG: +9.63% slower
...
```

### 4. Journal Summary Table
Ready-to-use untuk paper:
```csv
Metric,Non-RAG (LLM Standalone),RAG (LLM + RAG)
LLM Inference Time (mean),2.345s,2.567s
LLM Inference Time (std),0.123s,0.145s
Total Processing Time (mean),2.349s,2.575s
...
```

### 5. Visualizations
5 grafik PNG untuk paper/presentation:
- Response time comparison (box plots)
- Performance by category
- Time distribution histograms
- RAG document retrieval analysis
- Summary comparison bar chart

---

## 🔧 Troubleshooting

### Problem 1: Ollama Server Not Running

**Symptom:**
```
✗ Ollama server not running: Connection refused
```

**Solution:**
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Verify
curl http://127.0.0.1:11434/api/tags
```

---

### Problem 2: Model Not Found

**Symptom:**
```
⚠ Model llama3.2:3b not found
```

**Solution:**
```bash
# Pull model (requires internet - do this once)
ollama pull llama3.2:3b

# Verify
ollama list
```

---

### Problem 3: PDF Not Found

**Symptom:**
```
✗ referensi.pdf NOT found
```

**Solution:**
```bash
# Check if PDFs are in project directory
ls -la ~/voice_assistant_research/*.pdf

# Copy PDFs to correct location if needed
cp /path/to/referensi.pdf ~/voice_assistant_research/
cp /path/to/jurnal_tambahan.pdf ~/voice_assistant_research/
```

---

### Problem 4: Piper TTS Not Working

**Symptom:**
```
✗ Piper TTS not properly installed
```

**Solution:**
```bash
# Check Piper binary
ls -la /home/rangga/piper/build/piper

# Check model
ls -la /usr/local/share/piper/models/en_US-lessac-medium.onnx

# If missing, follow tutorial setup again
```

---

### Problem 5: Memory Issues

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
Enhanced assistant sudah dikonfigurasi untuk CPU mode. Jika masih ada masalah:

```python
# In enhanced_assistant.py, verify:
DEVICE = "cpu"  # Should be CPU, not cuda
```

---

### Problem 6: Slow Performance

**Tips untuk optimasi:**

1. **Reduce question set** (untuk quick test):
```python
USE_FULL_DATASET = False  # Use subset
```

2. **Reduce repetitions**:
```python
NUM_REPETITIONS = 1  # Instead of 3
```

3. **Reduce RAG top_k**:
```python
# In enhanced_assistant.py
RAG_TOP_K = 2  # Instead of 3
```

---

## 📝 Tips untuk Penelitian

### 1. Backup Results
```bash
# Setelah eksperimen selesai, backup hasil
cd ~/voice_assistant_research
tar -czf results_backup_$(date +%Y%m%d_%H%M%S).tar.gz experiment_results/
```

### 2. Multiple Runs
Untuk validasi, jalankan eksperimen beberapa kali di waktu berbeda:
```bash
# Morning run
python3 -c "import enhanced_assistant as ea; ea.run_text_test(...)"

# Afternoon run
python3 -c "import enhanced_assistant as ea; ea.run_text_test(...)"
```

### 3. Compare Different Configurations
Test dengan konfigurasi berbeda:
- Variasi jumlah dokumen retrieved (top_k)
- Variasi chunk size
- Variasi model (tiny vs base Whisper)

---

## 📚 Referensi

### Key Papers
- Tiny-Align paper (jurnal_tambahan.pdf)
- Tutorial reference (referensi.pdf)

### Documentation
- Ollama: https://ollama.ai/docs
- Whisper: https://github.com/openai/whisper
- FAISS: https://github.com/facebookresearch/faiss
- SentenceTransformers: https://www.sbert.net/

---

## ✅ Checklist Akhir

Sebelum memulai eksperimen, pastikan:

- [ ] Ollama server running (`ollama serve`)
- [ ] All dependencies installed
- [ ] PDFs (referensi.pdf, jurnal_tambahan.pdf) in place
- [ ] Jupyter notebook running
- [ ] Preparation notebook completed successfully
- [ ] Test mode works (`python3 enhanced_assistant.py test`)
- [ ] Backup space available (~500MB untuk results)

---

## 🎓 Untuk Jurnal

Data yang dihasilkan siap untuk:
1. **Methods section**: Jelaskan enhanced_assistant.py architecture
2. **Results section**: Gunakan plots dan summary tables
3. **Discussion**: Bandingkan dengan Tiny-Align paper results
4. **Conclusion**: RAG overhead vs quality improvement trade-off

---

## 📧 Support

Jika ada pertanyaan atau issues:
1. Check troubleshooting section
2. Review logs di `logs/` directory
3. Check Jupyter notebook output cells

---

**Good luck dengan penelitian Anda! 🚀**
