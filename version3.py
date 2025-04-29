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
import umap
from sklearn.neighbors import KernelDensity
import plotly.express as px
from sklearn.datasets import make_blobs

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
        st.dataframe(recent_dividends)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(recent_dividends.index, recent_dividends.values)
        ax.set_title("Dividend History")
        ax.set_xlabel("Date")
        ax.set_ylabel("Dividend ($)")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    st.subheader("Price History (Last 1 Year)")
    history = ticker_obj.history(period="1y")

    if history.empty:
        st.write("No price data available.")
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(history.index, history['Close'])
        ax.set_title("Price History")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price ($)")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    st.subheader("Key Financial Metrics")
    eps = info.get('trailingEps')
    dividend_rate = info.get('dividendRate')
    dividend_yield = info.get('dividendYield')

    payout_ratio = (dividend_rate / eps) if eps and dividend_rate else None

    st.write(f"Trailing EPS: {eps if eps else 'N/A'}")
    st.write(f"Dividend Rate: {dividend_rate if dividend_rate else 'N/A'}")
    st.write(f"Dividend Yield: {dividend_yield if dividend_yield else 'N/A'}")
    st.write(f"Dividend Payout Ratio: {round(payout_ratio,2) if payout_ratio else 'N/A'}")

# ============================================
# Altman Z-Score Functions
# ============================================

def compute_altman_z(ticker: str):
    ticker_obj = yf.Ticker(ticker)
    bs = ticker_obj.balance_sheet
    fs = ticker_obj.financials
    info = ticker_obj.info

    def fetch_balance(keys):
        for key in keys:
            for row in bs.index:
                if row.strip().lower() == key.strip().lower():
                    return bs.loc[row][0]
        return None

    def fetch_financials(keys):
        for key in keys:
            for row in fs.index:
                if row.strip().lower() == key.strip().lower():
                    return fs.loc[row][0]
        return None

    total_assets = fetch_balance(["Total Assets"])
    total_liabilities = fetch_balance(["Total Liabilities"])
    current_assets = fetch_balance(["Current Assets"])
    current_liabilities = fetch_balance(["Current Liabilities"])
    retained_earnings = fetch_balance(["Retained Earnings"])
    ebit = fetch_financials(["EBIT", "Operating Income"])
    sales = fetch_financials(["Total Revenue", "Sales"])

    share_price = info.get('regularMarketPrice')
    shares_outstanding = info.get('sharesOutstanding')
    market_cap = share_price * shares_outstanding if share_price and shares_outstanding else None

    if not all([total_assets, total_liabilities, market_cap]):
        return None, "Essential data missing."

    working_capital = (current_assets - current_liabilities) if current_assets and current_liabilities else 0
    ratio1 = working_capital / total_assets
    ratio2 = retained_earnings / total_assets
    ratio3 = ebit / total_assets
    ratio4 = market_cap / total_liabilities
    ratio5 = sales / total_assets

    z_score = 1.2*ratio1 + 1.4*ratio2 + 3.3*ratio3 + 0.6*ratio4 + 1.0*ratio5

    if z_score > 2.99:
        classification = "Safe Zone"
    elif z_score >= 1.81:
        classification = "Grey Zone"
    else:
        classification = "Distressed Zone"

    return z_score, classification

# ============================================
# Investing Analysis Functions
# ============================================

def investing_analysis():
    st.subheader("📈 Investing Analysis Coming Soon!")
    st.info("This section is under development.")

# ============================================
# Sector Density Explorer Functions
# ============================================

def display_sector_density():
    st.title("🌌 Sector Density Explorer")

    sectors = ['Technology', 'Finance', 'Healthcare', 'Energy', 'Industrial', 'Retail', 'Media', 'Transportation']
    X, y = make_blobs(n_samples=2000, centers=len(sectors), n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
    df['sector'] = [sectors[label] for label in y]

    reducer = umap.UMAP(n_components=3, random_state=42)
    embedding = reducer.fit_transform(df[['feature1', 'feature2', 'feature3', 'feature4', 'feature5']])
    df['x'] = embedding[:,0]
    df['y'] = embedding[:,1]
    df['z'] = embedding[:,2]

    kmeans = KMeans(n_clusters=20, random_state=42)
    df['cluster'] = kmeans.fit_predict(embedding)
    df['ticker'] = ['TICK' + str(i) for i in range(len(df))]

    query = st.text_input("Enter a Ticker to Explore (example: TICK100)")

    if query in df['ticker'].values:
        stock = df[df['ticker'] == query].iloc[0]
        st.info(f"{query} belongs to {stock['sector']} sector, Cluster #{stock['cluster']}.")

        df['distance'] = np.linalg.norm(df[['x','y','z']].values - stock[['x','y','z']].values, axis=1)
        nearby = df[df['ticker'] != query].sort_values('distance')

        def classify_tier(dist):
            if dist < 0.3: return 'Tier 1 🔴'
            elif dist < 0.6: return 'Tier 2 🟠'
            elif dist < 1.0: return 'Tier 3 🟡'
            else: return 'Tier 4 ⚪'

        nearby['Tier'] = nearby['distance'].apply(classify_tier)

        st.dataframe(nearby[['ticker', 'sector', 'cluster', 'distance', 'Tier']].head(10))

        fig = px.scatter_3d(
            df, x='x', y='y', z='z', color='sector',
            symbol=df['ticker'].apply(lambda t: 'star' if t == query else 'circle'),
            hover_data=['ticker','cluster']
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Ticker not found.")

# ============================================
# Main App
# ============================================

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dividend Dashboard", "Altman Z-Score", "Investing Analysis", "Sector Density Explorer"])

    if page == "Dividend Dashboard":
        ticker = st.text_input("Enter Stock Ticker", "AAPL")
        if st.button("Show Dividend Dashboard"):
            display_dividend_dashboard(ticker)

    elif page == "Altman Z-Score":
        ticker = st.text_input("Enter Stock Ticker", "AAPL", key='zscore')
        if st.button("Compute Altman Z-Score"):
            z_score, classification = compute_altman_z(ticker)
            if z_score:
                st.success(f"Altman Z-Score: {z_score:.2f} ({classification})")
            else:
                st.error(f"Error: {classification}")

    elif page == "Investing Analysis":
        investing_analysis()

    elif page == "Sector Density Explorer":
        display_sector_density()

if __name__ == "__main__":
    main()
