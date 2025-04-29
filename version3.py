# ============================================
# Import Libraries
# ============================================
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import requests
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
import umap
from sklearn.neighbors import KernelDensity
import plotly.express as px
from sklearn.datasets import make_blobs

# ============================================
# Sector Density Explorer Functions
# ============================================
def display_sector_density():
    st.title("🌌 Sector Density Explorer")

    # Simulated data setup
    sectors = ['Technology', 'Finance', 'Healthcare', 'Energy', 'Industrial', 'Retail', 'Media', 'Transportation']
    X, y = make_blobs(n_samples=2000, centers=len(sectors), n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
    df['sector'] = [sectors[label] for label in y]

    reducer = umap.UMAP(n_components=3, random_state=42)
    embedding = reducer.fit_transform(df[['feature1', 'feature2', 'feature3', 'feature4', 'feature5']])
    df['x'] = embedding[:, 0]
    df['y'] = embedding[:, 1]
    df['z'] = embedding[:, 2]

    kmeans = KMeans(n_clusters=20, random_state=42)
    df['cluster'] = kmeans.fit_predict(embedding)
    df['ticker'] = ['TICK' + str(i) for i in range(len(df))]

    # Sector selector
    sector_list = sorted(df['sector'].unique())
    selected_sectors = st.multiselect("Select up to 2 sectors to explore", sector_list, default=sector_list[:2])

    if selected_sectors:
        for sector in selected_sectors:
            sector_data = df[df['sector'] == sector]

            xyz = np.vstack([sector_data['x'], sector_data['y'], sector_data['z']]).T
            kde = KernelDensity(bandwidth=0.5, kernel='gaussian')
            kde.fit(xyz)
            density = np.exp(kde.score_samples(xyz))
            sector_data = sector_data.copy()
            sector_data['density'] = density

            fig = px.scatter_3d(
                sector_data,
                x='x', y='y', z='z',
                color='density',
                color_continuous_scale='YlOrRd',
                hover_data={
                    'ticker': True,
                    'cluster': True,
                    'density': ':.3f'
                },
                title=f"📊 {sector} Sector Density Map",
                width=1000,
                height=700
            )

            fig.update_traces(marker=dict(size=4, opacity=0.8))
            fig.update_layout(
                margin=dict(l=0, r=0, b=0, t=40),
                scene=dict(
                    xaxis_title="UMAP X",
                    yaxis_title="UMAP Y",
                    zaxis_title="UMAP Z"
                ),
                coloraxis_colorbar=dict(
                    title="Density",
                    tickformat=".2f"
                )
            )
            st.plotly_chart(fig, use_container_width=True)

        if len(selected_sectors) == 2:
            sec1, sec2 = selected_sectors
            data_sec1 = df[df['sector'] == sec1]
            data_sec2 = df[df['sector'] == sec2]
            common_clusters = set(data_sec1['cluster']).intersection(set(data_sec2['cluster']))
            hidden_competitors_sec1 = data_sec1[data_sec1['cluster'].isin(common_clusters)]
            hidden_competitors_sec2 = data_sec2[data_sec2['cluster'].isin(common_clusters)]

            st.subheader("💥 Hidden Competitors Detected")
            st.write(hidden_competitors_sec1[['ticker', 'cluster']])
            st.write(hidden_competitors_sec2[['ticker', 'cluster']])

    # --- Proximity Explorer ---
    st.markdown("### 🔍 Stock Proximity Explorer")
    query = st.text_input("Enter a Ticker (e.g., TICK123)")

    if query:
        if query in df['ticker'].values:
            stock = df[df['ticker'] == query].iloc[0]
            st.success(f"{query} is in **{stock['sector']}**, cluster #{stock['cluster']}.")

            df['distance'] = np.linalg.norm(df[['x', 'y', 'z']].values - stock[['x', 'y', 'z']].values, axis=1)
            nearby = df[df['ticker'] != query].sort_values('distance')

            def classify_tier(dist):
                if dist < 0.3:
                    return 'Tier 1 🔴'
                elif dist < 0.6:
                    return 'Tier 2 🟠'
                elif dist < 1.0:
                    return 'Tier 3 🟡'
                else:
                    return 'Tier 4 ⚪'

            nearby['Tier'] = nearby['distance'].apply(classify_tier)

            st.markdown("#### 📊 Closest Competitors")
            st.dataframe(nearby[['ticker', 'sector', 'cluster', 'distance', 'Tier']].head(10))

            fig = px.scatter_3d(
                df, x='x', y='y', z='z',
                color='sector',
                symbol=df['ticker'].apply(lambda t: 'star' if t == query else 'circle'),
                hover_data=['ticker', 'cluster'],
                title=f"🧽 Position of {query} Among Competitors"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("❌ Ticker not found. Try examples like TICK0, TICK42, TICK1987")

# ============================================
# Streamlit Main App
# ============================================
def main():
    st.title("Financial Dashboard")

    page = st.sidebar.radio(
        "Navigation", 
        ["Sector Density Explorer"]
    )

    if page == "Sector Density Explorer":
        display_sector_density()

if __name__ == "__main__":
    main()
