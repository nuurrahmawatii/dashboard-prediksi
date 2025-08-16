import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from utils import *

st.set_page_config(layout="wide")

# === CSS Styling ===
st.markdown("""
    <style>
        html, body, [data-testid="stApp"] {
            overflow: hidden !important;
            height: 100vh;
            background-color: #f2f2f2;
        }
        
        /* Menu kolom kiri */
        [data-testid="column"] :first-of-type{
            background-color: #c22323;
            padding: 1rem;
            border-radius: 10px;
            height: 100vh;
            box-sizing: border-box;
            box-shadow: 2px 0 8px rgba(0, 0, 0, 0.15);
            
        }

        /* Navbar */
        .navbar {
            background-color: #ac8b64;
            padding: 1rem 2rem;
            font-size: 20px;
            font-weight: bold;
            border-radius: 10px;
            margin-bottom: 1rem;
        }

        /* Hover: ubah warna latar dan teks */
        div.stButton > button:hover {
            background-color: #dcb991;
            color: white;
            border:none;
        }
    
        /* Button: hilangkan outline saat tombol difokuskan */
        div.stButton > button:focus,
        div.stButton > button:active{
            outline: none !important;
            box-shadow: none !important;
            border: none !important;
            background-color: #dcb991 !important; /* jaga tetap sama saat aktif */
            color: white !important;
            font-weight: bold;
        }
    
        /* Gaya normal tombol */
        div.stButton > button {
            box-shadow: 1px solid;
            border-radius: 8px;
            margin-top: 0.5rem;
            background-color: transparent;
            color: black;
            transition: all 0.2s ease;
        }
    </style>
""", unsafe_allow_html=True)

df = pd.read_csv('data/harga_kopi.csv')
df.set_index('Date', inplace=True)
data = df["Close"]

# ===== UI =====
st.set_page_config(page_title="Dashboard", layout="wide")
# === Navbar atas ===
st.markdown('<div class="navbar">Dashboard Prediksi Harga Kopi Berjangka (KC=F)</div>', unsafe_allow_html=True)

# Inisialisasi menu state
if "menu_state" not in st.session_state:
    st.session_state.menu_state = "Dashboard"

# === Layout fleksibel tergantung menu ===
if st.session_state.menu_state in ["Hasil Penelitian", "Dashboard"]:
    col_menu, col_content = st.columns([1, 5])

    # Kolom 1: Menu
    with col_menu:
        # st.markdown("### Menu")
        # if st.button("Hasil Penelitian", use_container_width=True):
        #     st.session_state.menu_state = "Hasil Penelitian"
        #     st.rerun()
        if st.button("Statistik Deskriptif", use_container_width=True):
            st.session_state.menu_state = "Statistik Deskriptif"
            st.rerun()
        if st.button("Evaluasi Model", use_container_width=True):
            st.session_state.menu_state = "Evaluasi Model"
            st.rerun()
        if st.button("Forecast", use_container_width=True):
            st.session_state.menu_state = "Forecast"
            st.rerun()

    # Kolom 2: Konten Dashboard dan Hasil Penelitian
    with col_content:
        if st.session_state.menu_state == "Dashboard":
            # st.subheader("Dashboard")
            st.markdown("""
                Dashboard ini menyajikan hasil penelitian skripsi mengenai **prediksi harga kopi berjangka**.
                Data yang digunakan pada penelitian ini adalah data penutupan harian harga kopi berjangka periode Januari 2004 hingga Desember 2023.
            """)
            st.markdown("""
                Prediksi dilakukan menggunakan metode _Long Short-Term Memory_, _Extreme Learning Machine_, dan _Hybrid_ LSTM-ELM.
                Proses pencarian hiperparameter terbaik dilakukan menggunakan metode _Particle Swarm Optimization_.
            """)
            st.markdown("""
                _**Catatan:** Data yang digunakan merupakan data perdagangan aktif pada hari kerja (Senin–Jumat).  
                Hari libur dan akhir pekan tidak termasuk dalam dataset karena tidak ada aktivitas pasar._
            """)
        # elif st.session_state.menu_state == "Hasil Penelitian":
        #    st.markdown("""
        #         ### Hasil Penelitian
        
        #         Hasil penelitian menunjukkan bahwa 
        #     """)
            

else:
    col_menu, col_plot, col_table = st.columns([1, 4, 1])

    # Kolom 1: Menu
    with col_menu:
        # st.markdown("### Menu")
        # if st.button("Hasil Penelitian", use_container_width=True):
        #     st.session_state.menu_state = "Hasil Penelitian"
        #     st.rerun()
        if st.button("Statistik Deskriptif", use_container_width=True):
            st.session_state.menu_state = "Statistik Deskriptif"
            st.rerun()
        if st.button("Evaluasi Model", use_container_width=True):
            st.session_state.menu_state = "Evaluasi Model"
            st.rerun()
        if st.button("Forecast", use_container_width=True):
            st.session_state.menu_state = "Forecast"
            st.rerun()
            
    # Kolom 3: Tabel
    with col_table:
        if st.session_state.menu_state == "Evaluasi Model":
            pilih_model_eval = st.selectbox(
                "Pilih Model",
                ["", "LSTM-PSO", "ELM-PSO", "LSTM-ELM-PSO", "ARIMA"],
                key="eval_model"
            )
    
            if pilih_model_eval:
                df_evaluasi, file_path = load_evaluation_from_csv(pilih_model_eval)
    
                if df_evaluasi is not None and not df_evaluasi.empty:
                    df_evaluasi = df_evaluasi[["RMSE", "MAE", "MAPE"]]  # Hanya tampilkan kolom yang kamu mau
                    st.markdown(f"### Hasil Evaluasi")
                    st.dataframe(df_evaluasi, use_container_width=True)
    
        elif st.session_state.menu_state == "Forecast":
            pilih_model_forecast = st.selectbox(
                "Pilih Model", ["", "LSTM-PSO", "ELM-PSO", "LSTM-ELM-PSO", "ARIMA"],
                key="forecast_model"
            )
            pilih_hari = st.selectbox("Jumlah hari", ["", "10", "15"], key="forecast_days")
    
            if pilih_model_forecast and pilih_hari:
                df_forecast, file_name = load_forecast_result(pilih_model_forecast, pilih_hari)
                if df_forecast is not None:
                    st.session_state["df_forecast"] = df_forecast
                    st.markdown("### Hasil Prediksi")
                    st.dataframe(df_forecast['Prediksi'], use_container_width=True)
                else:
                    st.warning(f"File `{file_name}` tidak ditemukan.")
    
        elif st.session_state.menu_state == "Statistik Deskriptif":
            st.write("Statistik Deskriptif")
            st.table(data.describe().round(2))
    
    
    # Kolom 2: Konten Plot / Visualisasi
    with col_plot:
        if st.session_state.menu_state == "Evaluasi Model":
            pilih_model_eval = st.session_state.get("eval_model", "")
    
            if pilih_model_eval:
                model_map = {
                    "LSTM-PSO": "lstm",
                    "ELM-PSO": "elm",
                    "LSTM-ELM-PSO": "hybrid",
                    "ARIMA": "arima"
                }
    
                nama_model = model_map.get(pilih_model_eval)
                if nama_model:
                    image = plot_evaluasi(nama_model)
                    if image:
                        st.image(image, caption=f"Plot Evaluasi: {pilih_model_eval}", use_container_width=True)
                    else:
                        st.warning(f"Plot gambar untuk model `{pilih_model_eval}` tidak ditemukan.")
            else:
                st.info("Silakan pilih model terlebih dahulu.")
    
        elif st.session_state.menu_state == "Forecast":
            st.subheader("Hasil Prediksi")
            df_forecast = st.session_state.get("df_forecast")
            pilih_model_forecast = st.session_state.get("forecast_model", "")
            pilih_hari = st.session_state.get("forecast_days", "")
    
            if df_forecast is not None and pilih_model_forecast and pilih_hari:
                df_hist = pd.read_csv("data/harga_kopi.csv", index_col=0, parse_dates=True)
                forecast_days = int(pilih_hari)
                forecast_vals = df_forecast["Prediksi"].values
    
                fig = plot_forecast(df_hist, forecast_vals, forecast_days,
                                    title=f"Forecast {pilih_model_forecast} - {pilih_hari} Hari")
                st.pyplot(fig)
            else:
                st.info("Silakan pilih model dan jumlah hari yang akan diprediksi.")
    
        elif st.session_state.menu_state == "Statistik Deskriptif":
            st.subheader("Dataset")
            st.line_chart(df['Close'])
