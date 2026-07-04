import streamlit as st
import joblib
import os
import re

st.set_page_config(
    page_title="UAS PBA - Analisis Ulasan TikTok", 
    page_icon="📱", 
    layout="wide"
)

st.title("📱 Aplikasi Aspect-Based Sentiment Analysis & NER Ulasan TikTok")
st.markdown("""
**Nama:** Maisie Jacinda  
**Program Studi:** Teknik Informatika - Universitas Dian Nuswantoro  
**Mata Kuliah:** Pemrosesan Bahasa Alami Berbasis Teks (Project-Based Learning)
""")
st.markdown("---")

MODEL_PATH = "models/absa_model.joblib"
VECTORIZER_PATH = "models/absa_vectorizer.joblib"

@st.cache_resource
def load_saved_models():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    return None, None

model_absa, vectorizer_absa = load_saved_models()

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

DAFTAR_ASPEK_APP = ["masuk", "daftar", "bug", "update", "perbarui", "jeda", "live", "lambat", "macet"]

if model_absa is None or vectorizer_absa is None:
    st.error("❌ Berkas model biner `.joblib` tidak ditemukan di folder `models/`!")
else:
    st.subheader("📥 Input Ulasan Pengguna")
    ulasan_baru = st.text_input(
        "Masukkan kalimat ulasan aplikasi TikTok yang ingin diuji:", 
        placeholder="Contoh: aplikasinya jelek setelah di update jadi lemot pas mau masuk live"
    )

    if ulasan_baru:
        st.markdown("### 🔄 Hasil Analisis Pemrosesan Sistem:")
        teks_terproses = bersihkan_teks_inputan(ulasan_baru)
    
        st.markdown("#### 🔍 1. Modul Named Entity Recognition (NER)")
        tokens_kalimat = teks_terproses.split()
        html_markup_ner = []
        aspek_ditemukan = []
        
        for token in tokens_kalimat:
            if token in DAFTAR_ASPEK_APP:
                html_markup_ner.append(f"<mark style='background-color: #FFFF00; padding: 2px 4px; border-radius: 4px;'><b>{token}</b> <small>[ASPECT]</small></mark>")
                aspek_ditemukan.append(token)
            else:
                html_markup_ner.append(token)
                
        st.markdown(f"**Visualisasi Token Level (BIO-Rules Check):** {' '.join(html_markup_ner)}", unsafe_allow_html=True)
    
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
                st.success(f"🟢 **Prediksi Polarity Sentimen:** {hasil_sentimen.upper()}")
            elif hasil_sentimen == "negatif":
                st.error(f"🔴 **Prediksi Polarity Sentimen:** {hasil_sentimen.upper()}")
            else:
                st.warning(f"🟡 **Prediksi Polarity Sentimen:** {hasil_sentimen.upper()}")

        with st.expander("🛠️ Lihat Alur Log Pemrosesan Data Teks (Jejak Berpikir Ilmiah)"):
            st.write(f"**1. Teks Input Asli:** `{ulasan_baru}`")
            st.write(f"**2. Hasil Preprocessing & Normalisasi Slang:** `{teks_terproses}`")
            st.write(f"**3. Dimensi Representasi Vektor Input (TF-IDF Matrix):** `{vektor_tfidf.shape}`")
