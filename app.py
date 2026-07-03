# app.py (Taruh langsung di luar folder/halaman depan repo GitHub)
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

st.title("📱 Aplikasi Aspect-Based Sentiment Analysis & NER Ulasan TikTok")
st.markdown("""
**Nama:** Maisie Jacinda  
**Program Studi:** Teknik Informatika - Universitas Dian Nuswantoro  
**Mata Kuliah:** Pemrosesan Bahasa Alami Berbasis Teks (Project-Based Learning)
""")
st.markdown("---")

# =====================================================================
# 2. MEMUAT MODEL BINER (.JOBLIB) 
# =====================================================================
MODEL_ABSA_PATH = "models/absa_model.joblib"
VECTORIZER_PATH = "models/absa_vectorizer.joblib"
MODEL_NER_PATH = "models/ner_crf_model.joblib"  # Path model NER berbasis CRF

@st.cache_resource
def load_saved_models():
    models = {}
    if os.path.exists(MODEL_ABSA_PATH) and os.path.exists(VECTORIZER_PATH):
        models['absa'] = joblib.load(MODEL_ABSA_PATH)
        models['vectorizer'] = joblib.load(VECTORIZER_PATH)
    else:
        models['absa'], models['vectorizer'] = None, None
        
    if os.path.exists(MODEL_NER_PATH):
        models['ner'] = joblib.load(MODEL_NER_PATH)
    else:
        models['ner'] = None
        
    return models

loaded_models = load_saved_models()
model_absa = loaded_models['absa']
vectorizer_absa = loaded_models['vectorizer']
model_ner = loaded_models['ner']

# =====================================================================
# 3. UTILITY FUNCTIONS (PREPROCESSING & CRF FEATURE EXTRACTION)
# =====================================================================
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

# Fungsi ekstraksi fitur untuk Model CRF (Harus sama dengan yang ada di notebook)
def token2features(sent, i):
    word = sent[i]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i-1]
        features.update({
            '-1:word.lower()': word1.lower(),
        })
    else:
        features['BOS'] = True  # Beginning of Sentence

    if i < len(sent) - 1:
        word1 = sent[i+1]
        features.update({
            '+1:word.lower()': word1.lower(),
        })
    else:
        features['EOS'] = True  # End of Sentence

    return features

def sent2features(sent):
    return [token2features(sent, i) for i in range(len(sent))]

# =====================================================================
# 4. PANEL INTERFACES DAN LOGIKA PREDIKSI
# =====================================================================
if model_absa is None or vectorizer_absa is None or model_ner is None:
    st.error("❌ Berkas model biner (`absa_model.joblib`, `absa_vectorizer.joblib`, atau `ner_crf_model.joblib`) tidak lengkap di folder `models/`!")
else:
    st.subheader("📥 Input Ulasan Pengguna")
    ulasan_baru = st.text_input(
        "Masukkan kalimat ulasan aplikasi TikTok yang ingin diuji:", 
        placeholder="Contoh: aplikasinya jelek setelah di update jadi lemot pas mau masuk live"
    )

    if ulasan_baru:
        st.markdown("### 🔄 Hasil Analisis Pemrosesan Sistem:")
        teks_terproses = bersihkan_teks_inputan(ulasan_baru)
        tokens_kalimat = teks_terproses.split()
        
        # -------------------------------------------------------------
        # FITUR 1: Named Entity Recognition (NER) dengan Machine Learning (CRF)
        # -------------------------------------------------------------
        st.markdown("#### 🔍 1. Modul Named Entity Recognition (NER)")
        
        # Ekstraksi fitur teks input dan prediksi tag menggunakan model CRF
        fitur_kalimat = sent2features(tokens_kalimat)
        prediksi_tags = model_ner.predict([fitur_kalimat])[0]
        
        html_markup_ner = []
        aspek_ditemukan = []
        
        # Mapping token dengan tag hasil prediksi model
        for token, tag in zip(tokens_kalimat, prediksi_tags):
            if tag != 'O':
                html_markup_ner.append(f"<mark style='background-color: #FFFF00; padding: 2px 4px; border-radius: 4px;'><b>{token}</b> <small>[{tag}]</small></mark>")
                aspek_ditemukan.append(token)
            else:
                html_markup_ner.append(token)
                
        st.markdown(f"**Visualisasi Token Level (Machine Learning CRF Prediction):** {' '.join(html_markup_ner)}", unsafe_allow_html=True)
        
        # -------------------------------------------------------------
        # FITUR 2: Aspect-Based Sentiment Analysis (ABSA) Classifier
        # -------------------------------------------------------------
        st.markdown("#### 📊 2. Modul Aspect-Based Sentiment Analysis (ABSA)")
        vektor_tfidf = vectorizer_absa.transform([teks_terproses])
        hasil_sentimen = model_absa.predict(vektor_tfidf)[0]
        
        kolom_kiri, kolom_kanan = st.columns(2)
        with kolom_kiri:
            if aspek_ditemukan:
                st.info(f"📍 **Aspek Keluhan Terdeteksi (Model ML):** {', '.join(set(aspek_ditemukan))}")
            else:
                st.info("📍 **Aspek Keluhan Terdeteksi (Model ML):** Umum / Tidak Spesifik")
                
        with kolom_kanan:
            if hasil_sentimen == "positif":
                st.success(f"🟢 **Prediksi Polarity Sentimen:** {hasil_sentimen.upper()}")
            elif hasil_sentimen == "negatif":
                st.error(f"🔴 **Prediksi Polarity Sentimen:** {hasil_sentimen.upper()}")
            else:
                st.warning(f"🟡 **Prediksi Polarity Sentimen:** {hasil_sentimen.upper()}")
                
        # -------------------------------------------------------------
        # FITUR 3: JEJAK BERPIKIR ILMIAH EXPANDER
        # -------------------------------------------------------------
        with st.expander("🛠️ Lihat Alur Log Pemrosesan Data Teks (Jejak Berpikir Ilmiah)"):
            st.write(f"**1. Teks Input Asli:** `{ulasan_baru}`")
            st.write(f"**2. Hasil Preprocessing & Normalisasi Slang:** `{teks_terproses}`")
            st.write(f"**3. Dimensi Representasi Vektor Klasifikasi Sentimen (TF-IDF Matrix):** `{vektor_tfidf.shape}`")
            st.write(f"**4. Sequence Tags yang Diprediksi Model CRF:** `{prediksi_tags}`")
