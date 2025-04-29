# ============================================
# Import Libraries
# ============================================
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
import plotly.express as px
import umap
from sklearn.datasets import make_blobs
from sklearn.neighbors import KernelDensity

# ============================================
# Custom Professional Styling (Dark Gold Theme)
# ============================================
st.markdown(
    """
    <style>
    .stApp { background-color: #111111; }
    h1, h2, h3, h4 { color: #FFD700; font-family: 'Open Sans', sans-serif; }
    .css-1d391kg { background-color: #111111; }
    p { font-family: 'Open Sans', sans-serif; color: #E0E0E0; }
    button { border: 2px solid #007BFF; border-radius: 10px; color: white; background-color: #111111; }
    button:hover { background-color: #007BFF; color: #FFFFFF; }
    input { background-color: #1A1A1A; color: #FFD700; }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================
# Dividend Dashboard Functions
# ============================================

def display_dividend_dashboard(ticker: str):
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info

    st.subheader("Company Overview")
    st.write(info.get('longBusinessSummary', "No overview available."))

    st.subheader("Dividend History (Last 10 Entries)")
    dividends = ticker_obj.dividends

    if dividends.empty:
        st.write("No dividend data available.")
    else:
        recent_dividends = dividends.tail(10)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(recent_dividends.index, recent_dividends.values)
        ax.set_title("Dividend History")
        ax.set_xlabel("Date")
        ax.set_ylabel("Dividend ($)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("Price History (Last 1 Year)")
    history = ticker_obj.history(period="1y")

    if not history.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(history.index, history['Close'], label="Close Price")
        ax.set_title("Price History")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price ($)")
        ax.legend()
        st.pyplot(fig)

    st.subheader("Key Financial Metrics")
    eps = info.get('trailingEps')
    dividend_rate = info.get('dividendRate')
    dividend_yield = info.get('dividendYield')
    payout_ratio = (dividend_rate / eps) if eps and dividend_rate else None

    st.write("Trailing EPS:", eps if eps else "N/A")
    st.write("Dividend Rate:", dividend_rate if dividend_rate else "N/A")
    st.write("Dividend Yield:", dividend_yield if dividend_yield else "N/A")
    st.write("Payout Ratio:", round(payout_ratio, 2) if payout_ratio else "N/A")

# ============================================
# Altman Z-Score Functions
# ============================================

def compute_altman_z(ticker: str):
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
    total_liabilities = fetch(["Total Liabilities", "Total Liabilities Net Minority Interest"], bs)
    current_assets = fetch(["Current Assets", "Total Current Assets"], bs)
    current_liabilities = fetch(["Current Liabilities", "Total Current Liabilities"], bs)
    retained_earnings = fetch(["Retained Earnings"], bs)
    ebit = fetch(["EBIT", "Operating Income"], fs)
    sales = fetch(["Total Revenue", "Sales"], fs)

    if None in [total_assets, total_liabilities]:
        return None, "Essential data missing."

    market_cap = info.get('regularMarketPrice', 0) * info.get('sharesOutstanding', 0)

    ratio1 = (current_assets - current_liabilities) / total_assets if current_assets and current_liabilities else 0
    ratio2 = retained_earnings / total_assets if retained_earnings else 0
    ratio3 = ebit / total_assets if ebit else 0
    ratio4 = market_cap / total_liabilities if total_liabilities else 0
    ratio5 = sales / total_assets if sales else 0

    z_score = 1.2*ratio1 + 1.4*ratio2 + 3.3*ratio3 + 0.6*ratio4 + ratio5

    if z_score > 2.99:
        classification = "Safe Zone"
    elif z_score >= 1.81:
        classification = "Grey Zone"
    else:
        classification = "Distressed Zone"

    return z_score, classification

# ============================================
# Sector Density Explorer (Improved)
# ============================================

def display_hidden_competitor_map(df):
    fig = px.scatter(
        df, x='x', y='y', color='sector', hover_data=['ticker', 'sector', 'cluster'],
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig.update_layout(title="Hidden Competitor Neural Map", title_font_size=22, height=700, width=1000)
    st.plotly_chart(fig, use_container_width=True)

def display_sector_density():
    st.title("Sector Density Explorer")

    sectors = ['Technology', 'Finance', 'Healthcare', 'Energy', 'Industrial', 'Retail', 'Media', 'Transportation']
    X, y = make_blobs(n_samples=2000, centers=len(sectors), n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
    df['sector'] = [sectors[label] for label in y]

    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(df[['feature1', 'feature2', 'feature3', 'feature4', 'feature5']])
    df['x'], df['y'] = embedding[:, 0], embedding[:, 1]

    kmeans = KMeans(n_clusters=20, random_state=42)
    df['cluster'] = kmeans.fit_predict(embedding)
    df['ticker'] = ['TICK' + str(i) for i in range(len(df))]

    selected_sectors = st.multiselect("Select up to 2 sectors", sorted(df['sector'].unique()), default=['Technology', 'Finance'])

    for sector in selected_sectors:
        sector_data = df[df['sector'] == sector]
        kde = KernelDensity(bandwidth=0.5).fit(sector_data[['x', 'y']])
        density = np.exp(kde.score_samples(sector_data[['x', 'y']]))
        sector_data = sector_data.copy()
        sector_data['density'] = density

        fig = px.scatter(
            sector_data, x='x', y='y', color='density', color_continuous_scale='Hot',
            hover_data=['ticker', 'cluster'], template="plotly_dark",
            title=f"{sector} Sector Density"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Full Hidden Competitor Neural Map")
    display_hidden_competitor_map(df)

# ============================================
# Streamlit App
# ============================================

def explain_backend():
    st.title("Explain Backend")
    st.write("This app analyzes dividends, bankruptcy risk (Z-Score), and sector competitor densities.")

def main():
    st.title("Financial Dashboard")

    page = st.sidebar.radio("Navigation", ["Dividend Dashboard", "Altman Z-Score", "Investing Analysis", "Sector Density Explorer", "Explain Backend"])

    if page == "Dividend Dashboard":
        ticker = st.text_input("Enter Ticker", "AAPL")
        if st.button("Show Dividend Info"):
            display_dividend_dashboard(ticker)

    elif page == "Altman Z-Score":
        ticker = st.text_input("Enter Ticker for Z-Score", "AAPL")
        if st.button("Compute Altman Z-Score"):
            result = compute_altman_z(ticker)
            if result[0] is not None:
                st.success(f"Altman Z-Score: {result[0]:.2f}")
                st.info(f"Classification: {result[1]}")
            else:
                st.error(result[1])

    elif page == "Investing Analysis":
        st.subheader("(Coming Soon)")

    elif page == "Sector Density Explorer":
        display_sector_density()

    elif page == "Explain Backend":
        explain_backend()

if __name__ == "__main__":
    main()
