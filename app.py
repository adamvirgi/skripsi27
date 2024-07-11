import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model dan scaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Judul aplikasi
st.title('Stunting Prediction App')

# Input data dari pengguna
usia = st.number_input('Umur (bulan)', min_value=0, max_value=10, value=2)
berat_badan = st.number_input('Berat Badan (kg)', min_value=0, max_value=50, value=12)
tinggi_badan = st.number_input('Tinggi Badan (cm)', min_value=0, max_value=150, value=80)
jenis_kelamin = st.selectbox('Jenis Kelamin', ['Laki-laki', 'Perempuan'])

# Konversi jenis kelamin ke numerik
jenis_kelamin = 1 if jenis_kelamin == 'Laki-laki' else 0

# Prediksi ketika tombol diklik
if st.button('Predict'):
    features = pd.DataFrame([[usia, berat_badan, tinggi_badan, jenis_kelamin]],
                            columns=['Umur (bulan)', 'Berat Badan (kg)', 'Tinggi Badan (cm)', 'Jenis Kelamin'])
    
    # Standarisasi fitur
    features = scaler.transform(features)
    
    # Prediksi
    prediction = model.predict(features)
    
    # Konversi prediksi ke kategori
    categories = {1: 'tinggi', 2: 'normal', 3: 'stunted', 4: 'severely stunted'}
    result = categories.get(prediction[0], 'Unknown')
    
    st.write(f'Hasil Prediksi: {result}')
