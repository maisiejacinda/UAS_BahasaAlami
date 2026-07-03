# app/app.py
import streamlit as st
import joblib
import os
import re

# =====================================================================
# 1. KONFIGURASI HALAMAN UTAMA (UI/UX)
# =====================================================================
st.set_page_config(
    page_title="UAS PBA - Analisis Ulasan TikTok", 
    page_icon="📱", 
    layout="wide"
)

# Judul Utama Aplikasi sesuai Tema Projek UAS
st.title("📱 Aplikasi Aspect-Based Sentiment Analysis & NER Ulasan TikTok")
st.markdown("""
**Nama:** Maisie Jacinda  
**Program Studi:** Teknik Informatika - Universitas Dian Nuswantoro  
**Mata Kuliah:** Pemrosesan Bahasa Alami Berbasis Teks (Project-Based Learning)
""")
st.markdown("---")

# =====================================================================
# 2. MEMUAT MODEL BINER (.JOBLIB) HASIL TRAINING
# =====================================================================
# Jalur folder disesuaikan dengan struktur repositori GitHub
MODEL_PATH = "models/absa_model.joblib"
VECTORIZER_PATH = "models/absa_vectorizer.joblib"

@st.cache_resource
def load_saved_models():
    """Fungsi untuk memuat model biner secara aman dari folder models/"""
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    return None, None

model_absa, vectorizer_absa = load_saved_models()

# =====================================================================
# 3. UTILITY FUNCTIONS (PREPROCESSING & REGEX) - SAMA DENGAN NOTEBOOK
# =====================================================================
# Kamus slang lokal yang sudah diuji pada Langkah 4 sebelumnya
KAMUS_SLANG_LOKAL = {
    "gk": "tidak", "ga": "tidak", "gak": "tidak",
    "udah": "sudah", "udh": "sudah",
    "lemot": "lambat", "ngeleg": "macet",
    "bgt": "sangat", "banget": "sangat", 
    "updat": "perbarui", "update": "perbarui"
}

def bersihkan_teks_inputan(text):
    """Pipeline pembersihan teks (Case Folding, Regex, & Normalisasi)"""
    if not text: 
        return ""
    
    # a. Case Folding
    text = text.lower()
    
    # b. Regex: Menghapus simbol bising, angka, dan spasi ganda
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # c. Normalisasi Slang kata per kata
    words = text.split()
    normalized_words = [KAMUS_SLANG_LOKAL[w] if w in KAMUS_SLANG_LOKAL else w for w in words]
    
    return " ".join(normalized_words)

# Daftar kata kunci komponen aspek (Rule-based NER) dari Langkah 9
DAFTAR_ASPEK_APP = ["masuk", "daftar", "bug", "update", "perbarui", "jeda", "live", "lambat", "macet"]

# =====================================================================
# 4. PANEL INTERFACES DAN LOGIKA PREDIKSI APPLIKASI
# =====================================================================
if model_absa is None or vectorizer_absa is None:
    st.error("❌ Berkas model biner `.joblib` tidak ditemukan di folder `models/`. Pastikan kamu sudah menjalankan file notebook training terlebih dahulu!")
else:
    # Form Input Teks komponen pengguna non-teknis
    st.subheader("📥 Input Ulasan Pengguna")
    ulasan_baru = st.text_input(
        "Masukkan kalimat ulasan aplikasi TikTok yang ingin diuji:", 
        placeholder="Contoh: aplikasinya jelek banget setelah di update jadi lemot pas mau masuk live"
    )

    if ulasan_baru:
        st.markdown("### 🔄 Hasil Analisis Pemrosesan Sistem:")
        
        # Jalankan Pembersihan Teks secara Runtut
        teks_terproses = bersihkan_teks_inputan(ulasan_baru)
        
        # -------------------------------------------------------------
        # FITUR 1: Named Entity Recognition (NER) Visualizer
        # -------------------------------------------------------------
        st.markdown("#### 🔍 1. Modul Named Entity Recognition (NER)")
        tokens_kalimat = teks_terproses.split()
        html_markup_ner = []
        aspek_ditemukan = []
        
        for token in tokens_kalimat:
            if token in DAFTAR_ASPEK_APP:
                # Memberikan highlight warna kuning cerah jika token termasuk Entitas Aspek
                html_markup_ner.append(f"<mark style='background-color: #FFFF00; padding: 2px 4px; border-radius: 4px;'><b>{token}</b> <small>[ASPECT]</small></mark>")
                aspek_ditemukan.append(token)
            else:
                html_markup_ner.append(token)
                
        st.markdown(f"**Visualisasi Token Level (BIO-Rules Check):** {' '.join(html_markup_ner)}", unsafe_allow_html=True)
        
        # -------------------------------------------------------------
        # FITUR 2: Aspect-Based Sentiment Analysis (ABSA) Classifier
        # -------------------------------------------------------------
        st.markdown("#### 📊 2. Modul Aspect-Based Sentiment Analysis (ABSA)")
        
        # Transformasi teks inputan ke bentuk vektor TF-IDF numerik
        vektor_tfidf = vectorizer_absa.transform([teks_terproses])
        
        # Prediksi kelas sentimen menggunakan kecerdasan buatan Naive Bayes hasil load
        hasil_sentimen = model_absa.predict(vektor_tfidf)[0]
        
        # Layout kolom untuk menampilkan info rangkuman aspek keluhan & klasifikasi sentimen
        kolom_kiri, kolom_kanan = st.columns(2)
        
        with kolom_kiri:
            if aspek_ditemukan:
                st.info(f"📍 **Aspek Keluhan Terdeteksi:** {', '.join(set(aspek_ditemukan))}")
            else:
                st.info("📍 **Aspek Keluhan Terdeteksi:** Umum / Keluhan Global")
                
        with kolom_kanan:
            if hasil_sentimen == "positif":
                st.success(f"🟢 **Prediksi Polarity Sentimen:** {hasil_sentimen.upper()}")
            elif hasil_sentimen == "negatif":
                st.error(f"🔴 **Prediksi Polarity Sentimen:** {hasil_sentimen.upper()}")
            else:
                st.warning(f"🟡 **Prediksi Polarity Sentimen:** {hasil_sentimen.upper()}")
                
        # -------------------------------------------------------------
        # FITUR 3: JEJAK BERPIKIR ILMIAH EXPANDER (UNTUT VERIFIKASI)
        # -------------------------------------------------------------
        with st.expander("🛠️ Lihat Alur Log Pemrosesan Data Teks (Jejak Berpikir Ilmiah)"):
            st.write(f"**1. Teks Input Asli:** `{ulasan_baru}`")
            st.write(f"**2. Hasil Preprocessing & Normalisasi Slang:** `{teks_terproses}`")
            st.write(f"**3. Dimensi Representasi Vektor Input (TF-IDF Matrix):** `{vektor_tfidf.shape}`")
