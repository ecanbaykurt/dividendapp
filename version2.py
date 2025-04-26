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

# ============================================
# Dividend Dashboard Functions
# ============================================

def display_dividend_dashboard(ticker: str):
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.fast_info if hasattr(ticker_obj, 'fast_info') else ticker_obj.info

    st.subheader("Company Overview")
    st.write(info.get('longBusinessSummary', "No overview available."))

    st.subheader("Dividend History (Last 10 Entries)")
    dividends = ticker_obj.dividends

    if dividends.empty:
        st.warning("No dividend data available for this ticker.")
    else:
        recent_dividends = dividends.tail(10)
        st.dataframe(recent_dividends)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(recent_dividends.index, recent_dividends.values, color='skyblue')
        ax.set_title("Dividend History (Last 10 Entries)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Dividend ($)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("Price History (Last 1 Year)")
    history = ticker_obj.history(period="1y")

    if history.empty:
        st.warning("No price data available for this ticker.")
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(history.index, history['Close'], label="Close Price", color='orange')
        ax.set_title("Price History (Last 1 Year)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price ($)")
        ax.legend()
        st.pyplot(fig)

    st.subheader("Key Financial Metrics")
    eps = info.get('trailingEps')
    dividend_rate = info.get('dividendRate')
    dividend_yield = info.get('dividendYield')
    payout_ratio = (dividend_rate / eps) if eps and dividend_rate else None

    metrics = {
        "Trailing EPS": eps,
        "Dividend Rate": dividend_rate,
        "Dividend Yield": dividend_yield,
        "Dividend Payout Ratio": round(payout_ratio, 2) if payout_ratio else None
    }
    st.json(metrics)

# ============================================
# Altman Z-Score Functions
# ============================================

def compute_altman_z(ticker: str):
    ticker_obj = yf.Ticker(ticker)
    bs = ticker_obj.balance_sheet
    fs = ticker_obj.financials
    info = ticker_obj.info

    def lookup(data, keys):
        for key in keys:
            for idx in data.index:
                if idx.strip().lower() == key.strip().lower():
                    return data.loc[idx][0]
        return None

    total_assets = lookup(bs, ["Total Assets"])
    total_liabilities = lookup(bs, ["Total Liabilities", "Total Liabilities Net Minority Interest"])
    current_assets = lookup(bs, ["Current Assets", "Total Current Assets"])
    current_liabilities = lookup(bs, ["Current Liabilities", "Total Current Liabilities"])
    retained_earnings = lookup(bs, ["Retained Earnings"])
    ebit = lookup(fs, ["EBIT", "Operating Income"])
    sales = lookup(fs, ["Total Revenue", "Sales"])

    share_price = info.get('regularMarketPrice')
    shares_outstanding = info.get('sharesOutstanding')
    market_cap = share_price * shares_outstanding if share_price and shares_outstanding else None

    if None in [total_assets, total_liabilities, market_cap]:
        return None, "Essential financial data missing."

    working_capital = (current_assets - current_liabilities) if current_assets and current_liabilities else 0
    ratios = [
        working_capital / total_assets,
        retained_earnings / total_assets,
        ebit / total_assets,
        market_cap / total_liabilities,
        sales / total_assets
    ]

    z_score = 1.2 * ratios[0] + 1.4 * ratios[1] + 3.3 * ratios[2] + 0.6 * ratios[3] + ratios[4]

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

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    soup = BeautifulSoup(requests.get(url).text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    return pd.read_html(str(table))[0]['Symbol'].tolist()

def batch_fetch_info(tickers):
    all_data = yf.Tickers(tickers)
    records = []

    for ticker, obj in all_data.tickers.items():
        try:
            info = obj.fast_info if hasattr(obj, 'fast_info') else obj.info
            records.append({
                'Ticker': ticker,
                'Dividend Yield': info.get('dividendYield', np.nan),
                'Expected Return': info.get('regularMarketPrice', np.nan),
                'Stability': info.get('beta', np.nan)
            })
        except Exception:
            records.append({'Ticker': ticker, 'Dividend Yield': np.nan, 'Expected Return': np.nan, 'Stability': np.nan})

    return pd.DataFrame(records)

def perform_clustering(df):
    df_clean = df.dropna(subset=['Dividend Yield', 'Expected Return', 'Stability'])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_clean[['Dividend Yield', 'Expected Return', 'Stability']])

    model = KMeans(n_clusters=3, random_state=42)
    df_clean['Cluster'] = model.fit_predict(scaled_features)

    return model, df_clean

def recommend_stocks(df, budget, model=None, preferences=None):
    df_clean = df.dropna(subset=['Dividend Yield', 'Expected Return', 'Stability'])

    if preferences:
        df_clean = df_clean.sort_values(preferences.get('priority', 'Dividend Yield'), ascending=False)

    if model:
        best_cluster = df_clean['Cluster'].mode()[0]
        df_clean = df_clean[df_clean['Cluster'] == best_cluster]

    selected = df_clean.head(5)
    selected['Allocation'] = budget / len(selected)

    return selected

# ============================================
# Streamlit App
# ============================================

def explain_backend():
    st.subheader("Backend Explanation")
    st.write(
        "This app uses Yahoo Finance and web scraping for real-time financial data, "
        "performs clustering on selected features for smarter stock picking, "
        "calculates Altman Z-Score for financial health checks, and speeds up batch data fetching."
    )

def main():
    st.title("Personalized Financial Dashboard")

    page = st.sidebar.radio("Navigation", ["Dividend Dashboard", "Altman Z-Score", "Investing Analysis", "Explain Backend"])
    user_preferences = {'priority': st.sidebar.selectbox("Investment Priority", ['Dividend Yield', 'Expected Return', 'Stability'])}

    if page == "Dividend Dashboard":
        ticker = st.text_input("Enter Ticker", "AAPL")
        if st.button("Show Dividend Info"):
            display_dividend_dashboard(ticker)

    elif page == "Altman Z-Score":
        ticker = st.text_input("Enter Ticker for Z-Score", "AAPL", key="zscore")
        if st.button("Compute Altman Z-Score"):
            z_score, classification = compute_altman_z(ticker)
            if z_score is not None:
                st.success(f"Altman Z-Score: {z_score:.2f}")
                st.info(f"Classification: {classification}")
            else:
                st.error(f"Error: {classification}")

    elif page == "Investing Analysis":
        budget = st.number_input("Investment Budget ($)", min_value=0)

        if st.button("Get Stock Recommendations"):
            with st.spinner("Fetching stock data and analyzing... please wait"):
                tickers = get_sp500_tickers()
                features_df = batch_fetch_info(tickers)
                model, clustered = perform_clustering(features_df)
                recommendations = recommend_stocks(clustered, budget, model=model, preferences=user_preferences)

            st.success("Analysis complete!")
            st.write("Top Recommended Stocks:")
            st.dataframe(recommendations)

    elif page == "Explain Backend":
        explain_backend()

if __name__ == "__main__":
    main()
