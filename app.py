
import streamlit as st
import pickle
import numpy as np

# ========================================
# Load model
# ========================================
model = pickle.load(open('model.pkl', 'rb'))

# ========================================
# Judul Halaman
# ========================================
st.title("🎯 Prediksi Penggunaan Layanan Makanan Online")
st.write("Aplikasi ini dibuat oleh **Willia Diva Ikhsani** untuk UAS MPML 2025.")

# ========================================
# Input fitur
# (Harus sesuai urutan saat training model)
# ========================================

age = st.number_input("Usia", min_value=10, max_value=100, step=1)
gender = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
marital_status = st.selectbox("Status Pernikahan", ["Belum Menikah", "Menikah"])
occupation = st.selectbox("Pekerjaan", ["Pelajar", "Mahasiswa", "Pegawai", "Wirausaha", "Lainnya"])
monthly_income = st.number_input("Penghasilan Bulanan (dalam ribu)", min_value=0)
education = st.selectbox("Pendidikan", ["SMA", "Diploma", "Sarjana", "Pascasarjana"])
family_size = st.slider("Jumlah anggota keluarga", 1, 10)
latitude = st.number_input("Latitude", format="%.6f")
longitude = st.number_input("Longitude", format="%.6f")
pin_code = st.number_input("Kode Pos", step=1)
feedback = st.slider("Feedback (0-10)", 0, 10)

# ========================================
# Encoding Manual (pastikan sesuai training)
# ========================================
gender_val = 0 if gender == "Perempuan" else 1
marital_val = 0 if marital_status == "Belum Menikah" else 1
occupation_val = {"Pelajar": 0, "Mahasiswa": 1, "Pegawai": 2, "Wirausaha": 3, "Lainnya": 4}[occupation]
education_val = {"SMA": 0, "Diploma": 1, "Sarjana": 2, "Pascasarjana": 3}[education]

# ========================================
# Gabungkan input ke array
# ========================================
input_data = np.array([[age, gender_val, marital_val, occupation_val, monthly_income,
                        education_val, family_size, latitude, longitude, pin_code, feedback]])

# ========================================
# Prediksi
# ========================================
if st.button("🔮 Prediksi"):
    prediction = model.predict(input_data)[0]
    result_text = "✅ Akan Memesan Online Food" if prediction == 1 else "❌ Tidak Akan Memesan"
    st.success(f"Hasil Prediksi: {result_text}")
