import streamlit as st
import joblib
import os
import re
# Mengimpor fungsi ekstraksi fitur CRF dari src/utils.py yang ada di repositorimu
from src.utils import sent2features 

st.set_page_config(
    page_title="UAS PBA - Analisis Ulasan TikTok", 
    page_icon="📱", 
    layout="wide"
)

st.title("📱 Aplikasi Aspect-Based Sentiment Analysis & NER Ulasan TikTok")
st.markdown("""
**Nama:** Maisie Jacinda  
**Program Studi:** Teknik Informatika - Universitas Dian Nuswantoro  
**NIM:** A11.2023.14985
**Mata Kuliah:** Pemrosesan Bahasa Alami Berbasis Teks (Project-Based Learning)
""")
st.markdown("---")

# Path menuju ketiga file model biner proyekmu
MODEL_ABSA_PATH = "models/absa_model.joblib"
VECTORIZER_PATH = "models/absa_vectorizer.joblib"
MODEL_NER_PATH = "models/ner_crf_model.joblib"

@st.cache_resource
def load_saved_models():
    # Memastikan ketiga file model terintegrasi utuh ke sistem
    if os.path.exists(MODEL_ABSA_PATH) and os.path.exists(VECTORIZER_PATH) and os.path.exists(MODEL_NER_PATH):
        model_absa = joblib.load(MODEL_ABSA_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        model_ner = joblib.load(MODEL_NER_PATH)
        return model_absa, vectorizer, model_ner
    return None, None, None

model_absa, vectorizer_absa, model_ner = load_saved_models()

KAMUS_SLANG_LOKAL = {
    "gk": "tidak", "ga": "tidak", "gak": "tidak",
    "udah": "sudah", "udh": "sudah",
    "lemot": "lambat", "ngeleg": "macet",
    "bgt": "sangat", "banget": "sangat", 
    "updat": "perbarui", "update": "perbarui"
}

def bersihkan_teks_inputan(text):
    if not text: 
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    words = text.split()
    normalized_words = [KAMUS_SLANG_LOKAL[w] if w in KAMUS_SLANG_LOKAL else w for w in words]
    return " ".join(normalized_words)

# Validasi kelengkapan berkas model di server Streamlit Cloud
if model_absa is None or vectorizer_absa is None or model_ner is None:
    st.error("❌ Berkas model biner `.joblib` tidak lengkap di folder `models/`! Pastikan absa_model, absa_vectorizer, dan ner_crf_model sudah ter-upload.")
else:
    st.subheader("📥 Input Ulasan Pengguna")
    ulasan_baru = st.text_input(
        "Masukkan kalimat ulasan aplikasi TikTok yang ingin diuji:", 
        placeholder="Contoh: aplikasinya jelek setelah di update jadi lemot pas mau masuk live"
    )

    if ulasan_baru:
        st.markdown("### 🔄 Hasil Analisis Pemrosesan Sistem:")
        teks_terproses = bersihkan_teks_inputan(ulasan_baru)
    
        st.markdown("#### 🔍 1. Modul Named Entity Recognition (NER berbasis ML CRF)")
        tokens_kalimat = teks_terproses.split()
        
        # 1. INTEGRASI UTUH MODEL NER CRF
        # Mengekstrak fitur token kalimat saat ini menggunakan fungsi dari src/utils.py
        fitur_kalimat = sent2features(tokens_kalimat)
        # Melakukan prediksi tag sekuensial (BIO Tagging) menggunakan model CRF
        prediksi_tags = model_ner.predict([fitur_kalimat])[0]
        
        html_markup_ner = []
        aspek_ditemukan = []
        
        # 2. PROSES VISUALISASI HASIL PREDIKSI CRF PADA INTERFACE
        for token, tag in zip(tokens_kalimat, prediksi_tags):
            if tag != 'O':  # Jika tag bernilai B-ASPECT atau I-ASPECT
                html_markup_ner.append(f"<mark style='background-color: #FFFF00; padding: 2px 4px; border-radius: 4px;'><b>{token}</b> <small>[{tag}]</small></mark>")
                aspek_ditemukan.append(token)
            else:
                html_markup_ner.append(token)
                
        st.markdown(f"**Visualisasi Token Level (CRF Machine Learning Output):** {' '.join(html_markup_ner)}", unsafe_allow_html=True)
    
        st.markdown("#### 📊 2. Modul Aspect-Based Sentiment Analysis (ABSA)")
        vektor_tfidf = vectorizer_absa.transform([teks_terproses])
        hasil_sentimen = model_absa.predict(vektor_tfidf)[0]
        
        kolom_kiri, kolom_kanan = st.columns(2)
        with kolom_kiri:
            if aspek_ditemukan:
                st.info(f"📍 **Aspek Keluhan Terdeteksi:** {', '.join(set(aspek_ditemukan))}")
            else:
                st.info("📍 **Aspek Keluhan Terdeteksi:** Umum")
                
        with kolom_kanan:
            if hasil_sentimen == "positif":
                st.success(f"🟢 **Prediksi Polarity Sentimen (ComplementNB):** {hasil_sentimen.upper()}")
            elif hasil_sentimen == "negatif":
                st.error(f"🔴 **Prediksi Polarity Sentimen (ComplementNB):** {hasil_sentimen.upper()}")
            else:
                st.warning(f"🟡 **Prediksi Polarity Sentimen (ComplementNB):** {hasil_sentimen.upper()}")

        with st.expander("🛠️ Lihat Alur Log Pemrosesan Data Teks (Jejak Berpikir Ilmiah)"):
            st.write(f"**1. Teks Input Asli:** `{ulasan_baru}`")
            st.write(f"**2. Hasil Preprocessing & Normalisasi Slang:** `{teks_terproses}`")
            st.write(f"**3. Urutan BIO Prediksi CRF:** `{prediksi_tags}`")
            st.write(f"**4. Dimensi Representasi Vektor Input (TF-IDF Matrix):** `{vektor_tfidf.shape}`")
