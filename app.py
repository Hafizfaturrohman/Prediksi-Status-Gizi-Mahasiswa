import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# ---------------------
# KONFIGURASI HALAMAN
# ---------------------
st.set_page_config(
    page_title="Prediksi Status Gizi Mahasiswa",
    page_icon="📈",
    layout="centered"
)

st.title("📈 - Prediksi Status Gizi Mahasiswa")
st.markdown("Prediksi status gizi berdasarkan pola aktivitas harian seperti kalori, protein, aktivitas fisik, dan tidur.")

# ---------------------
# DATASET CONTOH
# ---------------------
@st.cache_data
def load_data():
    return pd.DataFrame({
        'kalori': [2200, 1800, 2500, 3000, 1500, 2100, 2300, 1900, 2600, 2800,
                   1700, 2400, 1600, 2750, 1950, 3100, 1550, 2000, 2200, 2650],
        'protein': [70, 50, 80, 90, 40, 65, 75, 55, 85, 95,
                    45, 78, 38, 88, 52, 100, 42, 60, 68, 82],
        'aktivitas': ['Sedang', 'Ringan', 'Berat', 'Sedang', 'Sedentary',
                      'Sedang', 'Berat', 'Ringan', 'Berat', 'Sedang',
                      'Ringan', 'Berat', 'Sedentary', 'Sedang', 'Ringan',
                      'Berat', 'Sedentary', 'Sedang', 'Sedang', 'Berat'],
        'tidur': [7, 6, 5, 8, 4, 7, 6, 6, 5, 8, 5, 7, 4, 8, 6, 9, 4, 7, 6, 5],
        'status_gizi': ['Normal', 'Kurang', 'Normal', 'Lebih', 'Kurang',
                        'Normal', 'Normal', 'Kurang', 'Normal', 'Lebih',
                        'Kurang', 'Normal', 'Kurang', 'Lebih', 'Kurang',
                        'Lebih', 'Kurang', 'Normal', 'Normal', 'Normal']
    })

df = load_data()

# ---------------------
# TRAINING MODEL
# ---------------------
df_encoded = pd.get_dummies(df, columns=['aktivitas'])
X = df_encoded.drop(columns='status_gizi')
y = df_encoded['status_gizi']

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

# ---------------------
# TABS
# ---------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Prediksi", "🌳 Visualisasi Model", "📁 Dataset", "ℹ️ Tentang Model"
])

# ---------------------
# TAB 1: PREDIKSI
# ---------------------
with tab1:
    st.subheader("Prediksi Status Gizi Mahasiswa")

    with st.form("form_prediksi"):
        col1, col2 = st.columns(2)
        with col1:
            kalori = st.number_input("Asupan Kalori (kcal)", 1000, 5000, 2200)
            protein = st.number_input("Asupan Protein (gr)", 0, 200, 60)
        with col2:
            aktivitas = st.selectbox("Aktivitas Fisik", ["Sedentary", "Ringan", "Sedang", "Berat"])
            tidur = st.slider("Durasi Tidur (jam)", 3, 12, 7)

        prediksi = st.form_submit_button("🔍 Prediksi Sekarang")

    if prediksi:
        input_data = pd.DataFrame([{
            'kalori': kalori,
            'protein': protein,
            'tidur': tidur,
            'aktivitas_Berat': 1 if aktivitas == "Berat" else 0,
            'aktivitas_Ringan': 1 if aktivitas == "Ringan" else 0,
            'aktivitas_Sedang': 1 if aktivitas == "Sedang" else 0,
            'aktivitas_Sedentary': 1 if aktivitas == "Sedentary" else 0
        }])

        hasil = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]

        st.success(f"Prediksi Status Gizi: **{hasil}**")

        proba_df = pd.DataFrame({
            'Status Gizi': model.classes_,
            'Probabilitas': proba
        }).sort_values("Probabilitas", ascending=False)

        st.subheader("Probabilitas Prediksi")
        st.bar_chart(proba_df.set_index("Status Gizi"))

# ---------------------
# TAB 2: VISUALISASI MODEL
# ---------------------
with tab2:
    st.subheader("Struktur Decision Tree")
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_tree(model, feature_names=X.columns, class_names=model.classes_,
              filled=True, rounded=True, fontsize=10, ax=ax)
    st.pyplot(fig)

# ---------------------
# TAB 3: DATASET
# ---------------------
with tab3:
    st.subheader("Dataset Mahasiswa")
    st.dataframe(df, use_container_width=True)

# ---------------------
# TAB 4: TENTANG MODEL
# ---------------------
with tab4:
    st.subheader("Tentang Model Prediksi")
    st.markdown("""
    Aplikasi **NutriTrack** menggunakan model *Decision Tree Classifier* untuk memprediksi status gizi mahasiswa berdasarkan:
    - Asupan kalori dan protein harian
    - Durasi tidur
    - Tingkat aktivitas fisik

    Dataset yang digunakan adalah data simulasi mahasiswa sebanyak 20 sampel, dengan label status gizi:
    - **Kurang**
    - **Normal**
    - **Lebih**

    Tujuan dari model ini adalah memberikan gambaran awal mengenai pola hidup yang mungkin berpengaruh pada status gizi mahasiswa.
    """)
