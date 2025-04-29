# ============================================
# Import Libraries
# ============================================
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.neighbors import KernelDensity
from scipy import stats
import requests
from bs4 import BeautifulSoup
import umap
import plotly.express as px

# ============================================
# Custom Futuristic Styling
# ============================================
st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #0A0A0A;
        color: #E0E0E0;
        font-family: 'sans-serif';
    }
    .css-1d391kg {
        background-color: #0A0A0A;
    }
    .stButton>button {
        border: 2px solid #00FFFF;
        border-radius: 8px;
        color: white;
        background-color: #0A0A0A;
    }
    .stButton>button:hover {
        background-color: #00FFFF;
        color: black;
    }
    input {
        background-color: #1A1A1A;
        color: #00FFFF;
    }
    h1, h2, h3 {
        color: #00FFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================
# Pages
# ============================================

# Home Page
def home_page():
    st.title("🚀 Welcome to the Future of Financial Intelligence")
    st.markdown("Smart, Fast, Predictive Insights for Modern Investors.")
    st.markdown("---")

# About Page
def about_page():
    st.title("About This App")
    st.write("""
        This dashboard helps you explore financial metrics, bankruptcy risks, 
        investing clusters, and competitor densities using modern AI-driven visuals.
        Built with ❤️ for next-generation investors and analysts.
    """)

# Dividend Dashboard
def display_dividend_dashboard():
    ticker = st.text_input("Enter Ticker Symbol:", "AAPL")
    if ticker:
        t_obj = yf.Ticker(ticker)
        info = t_obj.info
        dividends = t_obj.dividends

        st.subheader("Company Overview")
        st.write(info.get('longBusinessSummary', "No overview available."))

        st.subheader("Dividend History")
        if dividends.empty:
            st.write("No dividend history available.")
        else:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(dividends.tail(10).index, dividends.tail(10).values)
            ax.set_title("Recent Dividends")
            st.pyplot(fig)

# Altman Z-Score
def compute_altman_z():
    ticker = st.text_input("Enter Ticker for Z-Score:", "AAPL")
    if ticker:
        t_obj = yf.Ticker(ticker)
        bs = t_obj.balance_sheet
        fs = t_obj.financials
        info = t_obj.info

        def fetch(keys, df):
            for key in keys:
                for row in df.index:
                    if row.strip().lower() == key.strip().lower():
                        return df.loc[row][0]
            return None

        total_assets = fetch(["Total Assets"], bs)
        total_liabilities = fetch(["Total Liabilities"], bs)
        retained_earnings = fetch(["Retained Earnings"], bs)
        ebit = fetch(["EBIT", "Operating Income"], fs)
        sales = fetch(["Total Revenue", "Sales"], fs)

        if None in [total_assets, total_liabilities]:
            st.error("Essential financial data missing.")
            return

        market_cap = info.get('regularMarketPrice', 0) * info.get('sharesOutstanding', 0)
        ratio1 = (fetch(["Current Assets"], bs) - fetch(["Current Liabilities"], bs)) / total_assets if fetch(["Current Assets"], bs) and fetch(["Current Liabilities"], bs) else 0
        ratio2 = retained_earnings / total_assets if retained_earnings else 0
        ratio3 = ebit / total_assets if ebit else 0
        ratio4 = market_cap / total_liabilities if total_liabilities else 0
        ratio5 = sales / total_assets if sales else 0

        z_score = 1.2*ratio1 + 1.4*ratio2 + 3.3*ratio3 + 0.6*ratio4 + ratio5

        st.success(f"Altman Z-Score: {z_score:.2f}")
        if z_score > 2.99:
            st.info("Safe Zone ✅")
        elif z_score >= 1.81:
            st.warning("Grey Zone ⚠️")
        else:
            st.error("Distress Zone 🚨")

# Investing Analysis (Simple placeholder for now)
def investing_analysis():
    st.header("Investing Analysis Coming Soon 📈")
    st.info("Feature under development... Stay tuned!")

# Sector Density Explorer
def display_sector_density():
    st.title("Sector Density Explorer")

    sectors = ['Technology', 'Finance', 'Healthcare', 'Energy', 'Industrial', 'Retail', 'Media', 'Transportation']
    X, y = make_blobs(n_samples=2000, centers=len(sectors), n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=[f"feature{i}" for i in range(1, 6)])
    df['sector'] = [sectors[label] for label in y]

    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(df[[f"feature{i}" for i in range(1, 6)]])
    df['x'], df['y'] = embedding[:, 0], embedding[:, 1]

    fig = px.scatter(
        df, x='x', y='y', color='sector',
        hover_data=['sector'], template='plotly_dark',
        title="Hidden Competitor Neural Map"
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# Main App
# ============================================
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to:",
        ["Home", "Dividend Dashboard", "Altman Z-Score", "Investing Analysis", "Sector Density Explorer", "About This App"]
    )

    if page == "Home":
        home_page()
    elif page == "Dividend Dashboard":
        display_dividend_dashboard()
    elif page == "Altman Z-Score":
        compute_altman_z()
    elif page == "Investing Analysis":
        investing_analysis()
    elif page == "Sector Density Explorer":
        display_sector_density()
    elif page == "About This App":
        about_page()

if __name__ == "__main__":
    main()
