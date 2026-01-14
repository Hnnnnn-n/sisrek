import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df = pd.read_csv("tfidf_lagu.csv")

# Ambil kolom
titles = df["title"]
artists = df["artist"]
tfidf_matrix = df.iloc[:, 2:].values  # TF-IDF mulai kolom ke-2

st.title("Sistem Rekomendasi Musik 🎵")

# Pilih lagu
selected_title = st.selectbox("Pilih Lagu:", titles)

if st.button("Rekomendasikan"):
    # Cari index lagu
    idx = titles[titles == selected_title].index[0]

    # Hitung cosine similarity
    sim_scores = cosine_similarity([tfidf_matrix[idx]], tfidf_matrix)[0]

    # Ambil top 10 selain dirinya sendiri
    top_indices = np.argsort(sim_scores)[::-1][1:11]

    st.subheader("Rekomendasi Lagu:")
    for i in top_indices:
        st.write(f"**{titles[i]}** - {artists[i]} (Similarity: {sim_scores[i]:.4f})")
