import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import streamlit as st

# Fungsi untuk memuat data dan model
def load_data():
    # Ganti dengan path dataset Anda
    df = pd.read_csv('kurs rupiah to dollar.csv')
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%m/%d/%Y')
    return df

def train_model(df):
    # Data preprocessing dan feature engineering
    X = df['Tanggal'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    y = df['Kurs Jual'].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Membuat model MLPRegressor
    model = MLPRegressor(hidden_layer_sizes=(16, 16), max_iter=1000)
    model.fit(X_train, y_train)

    # Menyimpan model ke file
    joblib.dump(model, 'model.pkl')

    # Prediksi pada data uji
    y_pred = model.predict(X_test)
    return X_test, y_test, y_pred

def plot_results(df, X_test, y_test, y_pred):
    # Membuat figure dan axes
    fig, ax = plt.subplots(figsize=(12, 6))

    # Menambahkan plot ke axes
    ax.plot(df['Tanggal'][-len(y_test):], y_test, label='Actual', color='blue', marker='o', linestyle='-', markersize=5)
    ax.plot(df['Tanggal'][-len(y_pred):], y_pred, label='Predicted', color='red', marker='x', linestyle='--', markersize=5)

    # Menambahkan legenda dan label
    ax.legend()
    ax.set_title('Actual vs Predicted Kurs Jual')
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Kurs Jual')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Menampilkan plot di Streamlit
    st.pyplot(fig)

# Streamlit interface
st.title('Prediksi Kurs Jual Rupiah terhadap USD')
st.write("Aplikasi ini digunakan untuk memprediksi kurs jual Rupiah terhadap USD")

# Memuat data dan melatih model
df = load_data()

# Pilihan untuk melatih model atau memuat model yang sudah ada
if st.button('Train Model'):
    X_test, y_test, y_pred = train_model(df)
    st.write("Model trained successfully!")
    plot_results(df, X_test, y_test, y_pred)
elif st.button('Load Model & Predict'):
    # Memuat model yang sudah disimpan
    model = joblib.load('model.pkl')

    # Melakukan prediksi dengan model yang sudah dilatih
    X = df['Tanggal'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    y_pred_loaded_model = model.predict(X)

    st.write("Predictions loaded from model!")
    plot_results(df, X, df['Kurs Jual'], y_pred_loaded_model)
