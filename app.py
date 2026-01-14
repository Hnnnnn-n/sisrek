import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df = pd.read_csv("tfidf_lagu.csv")

titles = df["title"]
artists = df["artist"]
tfidf_matrix = df.iloc[:, 2:].values

st.set_page_config(page_title="Rekomendasi Musik", layout="wide")

st.title("🎵 Sistem Rekomendasi Musik")

selected_title = st.selectbox("Pilih Lagu:", titles)

# Simpan halaman aktif
if "page" not in st.session_state:
    st.session_state.page = 1

if st.button("Rekomendasikan"):
    st.session_state.page = 1  # reset ke halaman 1 saat lagu diganti
    st.session_state.run = True

if "run" in st.session_state and st.session_state.run:

    idx = titles[titles == selected_title].index[0]
    sim_scores = cosine_similarity([tfidf_matrix[idx]], tfidf_matrix)[0]

    # Ambil semua rekomendasi (tanpa dirinya sendiri)
    sorted_indices = np.argsort(sim_scores)[::-1][1:]

    # Pagination setting
    items_per_page = 8
    total_items = len(sorted_indices)
    total_pages = (total_items // items_per_page) + 1

    start = (st.session_state.page - 1) * items_per_page
    end = start + items_per_page
    page_indices = sorted_indices[start:end]

    st.subheader("Hasil Rekomendasi")

    # CSS Card Grid
    st.markdown("""
        <style>
        .grid-container {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 16px;
        }
        .card {
            background: #ffffff;
            padding: 14px;
            border-radius: 12px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        }
        .title {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 4px;
            color: #222;
        }
        .artist {
            font-size: 14px;
            color: #666;
            margin-bottom: 6px;
        }
        .score {
            font-size: 12px;
            color: #999;
        }
        </style>
    """, unsafe_allow_html=True)

    # Render Card
    cards_html = "<div class='grid-container'>"
    for i in page_indices:
        cards_html += f"""
        <div class='card'>
            <div class='title'>{titles[i]}</div>
            <div class='artist'>{artists[i]}</div>
            <div class='score'>Similarity: {sim_scores[i]:.4f}</div>
        </div>
        """
    cards_html += "</div>"

    st.markdown(cards_html, unsafe_allow_html=True)

    # --- Pagination Buttons ---
    st.write("Halaman:")
    cols = st.columns(total_pages)

    for i in range(total_pages):
        if cols[i].button(str(i+1)):
            st.session_state.page = i+1
            st.rerun()
