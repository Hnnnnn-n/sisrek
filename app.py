import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df = pd.read_csv("tfidf_lagu.csv")

titles = df["title"]
artists = df["artist"]
tfidf_matrix = df.iloc[:, 2:].values  # TF-IDF mulai kolom ke-2

st.set_page_config(page_title="Rekomendasi Musik", layout="wide")

st.title("🎵 Sistem Rekomendasi Musik")

selected_title = st.selectbox("Pilih Lagu:", titles)

if st.button("Rekomendasikan"):
    idx = titles[titles == selected_title].index[0]

    sim_scores = cosine_similarity([tfidf_matrix[idx]], tfidf_matrix)[0]
    top_indices = np.argsort(sim_scores)[::-1][1:13]  # ambil 12 lagu

    st.subheader("Hasil Rekomendasi")

    # CSS untuk card
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
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .card:hover {
            transform: translateY(-4px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
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

    # Buat grid card
    cards_html = "<div class='grid-container'>"
    for i in top_indices:
        cards_html += f"""
        <div class='card'>
            <div class='title'>{titles[i]}</div>
            <div class='artist'>{artists[i]}</div>
            <div class='score'>Similarity: {sim_scores[i]:.4f}</div>
        </div>
        """
    cards_html += "</div>"

    st.markdown(cards_html, unsafe_allow_html=True)
