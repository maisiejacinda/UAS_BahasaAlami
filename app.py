import streamlit as st
import joblib
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

st.set_page_config(
    page_title="UAS PBA - Analisis Ulasan TikTok",
    page_icon="📱",
    layout="wide"
)

st.title("📱 Aplikasi Aspect-Based Sentiment Analysis & NER Ulasan TikTok")
st.markdown("""
**Nama:** Maisie Jacinda  
**Program Studi:** Teknik Informatasi - Universitas Dian Nuswantoro  
**NIM:** A11.2023.14985  
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
    "gk": "tidak",
    "ga": "tidak",
    "gak": "tidak",
    "udah": "sudah",
    "udh": "sudah",
    "lemot": "lambat",
    "ngeleg": "macet",
    "bgt": "sangat",
    "banget": "sangat",
    "updat": "perbarui",
    "update": "perbarui"
}

def bersihkan_teks_inputan(text):
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()

    normalized_words = [
        KAMUS_SLANG_LOKAL[w] if w in KAMUS_SLANG_LOKAL else w
        for w in words
    ]

    return " ".join(normalized_words)

DAFTAR_ASPEK_APP = [
    "masuk",
    "daftar",
    "bug",
    "update",
    "perbarui",
    "jeda",
    "live",
    "lambat",
    "macet"
]

if model_absa is None or vectorizer_absa is None:
    st.error("❌ Berkas model biner `.joblib` tidak ditemukan di folder `models/`!")

else:

    st.subheader("📥 Input Ulasan Pengguna")

    ulasan_baru = st.text_input(
        "Masukkan kalimat ulasan aplikasi TikTok yang ingin diuji:",
        placeholder="Contoh: aplikasinya jelek setelah di update jadi lemot pas mau masuk live"
    )

    st.markdown("### 📂 Upload File CSV")

    uploaded_file = st.file_uploader(
        "Upload file CSV yang memiliki kolom 'ulasan'",
        type=["csv"]
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

                html_markup_ner.append(
                    f"<mark style='background-color:#FFFF00;padding:2px 4px;border-radius:4px;'><b>{token}</b> <small>[ASPECT]</small></mark>"
                )

                aspek_ditemukan.append(token)

            else:

                html_markup_ner.append(token)

        st.markdown(
            f"**Visualisasi Token Level (BIO-Rules Check):** {' '.join(html_markup_ner)}",
            unsafe_allow_html=True
        )

        st.markdown("#### 📊 2. Modul Aspect-Based Sentiment Analysis (ABSA)")

        vektor_tfidf = vectorizer_absa.transform([teks_terproses])

        hasil_sentimen = model_absa.predict(vektor_tfidf)[0]

        kolom_kiri, kolom_kanan = st.columns(2)

        with kolom_kiri:

            if aspek_ditemukan:

                st.info(
                    f"📍 **Aspek Keluhan Terdeteksi:** {', '.join(set(aspek_ditemukan))}"
                )

            else:

                st.info("📍 **Aspek Keluhan Terdeteksi:** Umum")

        with kolom_kanan:

            if hasil_sentimen == "positif":

                st.success(
                    f"🟢 **Prediksi Polarity Sentimen:** {hasil_sentimen.upper()}"
                )

            elif hasil_sentimen == "negatif":

                st.error(
                    f"🔴 **Prediksi Polarity Sentimen:** {hasil_sentimen.upper()}"
                )

            else:

                st.warning(
                    f"🟡 **Prediksi Polarity Sentimen:** {hasil_sentimen.upper()}"
                )

        with st.expander("🛠️ Lihat Alur Log Pemrosesan Data Teks (Jejak Berpikir Ilmiah)"):

            st.write(f"**1. Teks Input Asli:** `{ulasan_baru}`")
            st.write(f"**2. Hasil Preprocessing & Normalisasi Slang:** `{teks_terproses}`")
            st.write(f"**3. Dimensi Representasi Vektor Input (TF-IDF Matrix):** `{vektor_tfidf.shape}`")

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)

            if "ulasan" not in df.columns:
                st.error("❌ File CSV harus memiliki kolom bernama 'ulasan'")

            else:

                hasil_sentimen = []
                hasil_aspek = []

                for teks in df["ulasan"]:

                    teks = str(teks)

                    teks_bersih = bersihkan_teks_inputan(teks)

                    vector = vectorizer_absa.transform([teks_bersih])

                    prediksi = model_absa.predict(vector)[0]

                    aspek = []

                    for token in teks_bersih.split():

                        if token in DAFTAR_ASPEK_APP:
                            aspek.append(token)

                    hasil_sentimen.append(prediksi)

                    if len(aspek) == 0:
                        hasil_aspek.append("Umum")
                    else:
                        hasil_aspek.append(", ".join(sorted(set(aspek))))

                df["Sentimen"] = hasil_sentimen
                df["Aspek"] = hasil_aspek

                st.success("✅ Analisis CSV berhasil dilakukan.")

                st.subheader("📋 Hasil Analisis")

                st.dataframe(df, use_container_width=True)

                csv = df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="⬇️ Download Hasil Analisis",
                    data=csv,
                    file_name="hasil_analisis_tiktok.csv",
                    mime="text/csv"
                )

                st.markdown("---")
                st.header("📊 Dashboard Evaluasi")

                col1, col2 = st.columns(2)

                sentiment_counts = df["Sentimen"].value_counts()

                with col1:

                    st.subheader("Distribusi Sentimen")

                    st.bar_chart(sentiment_counts)

                # -----------------------------
                # Pie Chart
                # -----------------------------

                with col2:

                    st.subheader("Persentase Sentimen")

                    fig, ax = plt.subplots(figsize=(5,5))

                    ax.pie(
                        sentiment_counts.values,
                        labels=sentiment_counts.index,
                        autopct="%1.1f%%",
                        startangle=90
                    )

                    ax.axis("equal")

                    st.pyplot(fig)

                # -----------------------------
                # Statistik
                # -----------------------------

                st.subheader("📈 Ringkasan Data")

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Jumlah Data",
                    len(df)
                )

                c2.metric(
                    "Jumlah Sentimen",
                    len(sentiment_counts)
                )

                c3.metric(
                    "Aspek Unik",
                    len(set(df["Aspek"]))
                )

                semua_aspek = []

                for aspek in df["Aspek"]:

                    if aspek != "Umum":

                        semua_aspek.extend(aspek.split(", "))

                if len(semua_aspek) > 0:

                    counter = Counter(semua_aspek)

                    aspek_df = pd.DataFrame({
                        "Aspek": counter.keys(),
                        "Jumlah": counter.values()
                    })

                    st.subheader("📍 Aspek yang Paling Banyak Muncul")

                    st.bar_chart(
                        aspek_df.set_index("Aspek")
                    )

                    st.dataframe(
                        aspek_df,
                        use_container_width=True
                    )

                else:

                    st.info("Tidak ditemukan aspek khusus pada data CSV.")

        except Exception as e:

            st.error(f"Terjadi kesalahan saat membaca file CSV.\n\n{e}")
