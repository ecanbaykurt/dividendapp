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
        st.write("No dividend data available for this ticker.")
    else:
        recent_dividends = dividends.tail(10)
        st.write(recent_dividends)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(recent_dividends.index, recent_dividends.values)
        ax.set_title("Dividend History (Last 10 Entries)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Dividend ($)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("Price History (Last 1 Year)")
    history = ticker_obj.history(period="1y")

    if history.empty:
        st.write("No price data available for this ticker.")
    else:
        st.write(history[['Close']].head())

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(history.index, history['Close'], label="Close Price")
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

    st.write("Trailing EPS:", eps if eps is not None else "N/A")
    st.write("Dividend Rate:", dividend_rate if dividend_rate is not None else "N/A")
    st.write("Dividend Yield:", dividend_yield if dividend_yield is not None else "N/A")
    st.write("Dividend Payout Ratio:", round(payout_ratio, 2) if payout_ratio else "N/A")

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
    total_liabilities = fetch_balance(["Total Liabilities", "Total Liabilities Net Minority Interest"])
    current_assets = fetch_balance(["Current Assets", "Total Current Assets"])
    current_liabilities = fetch_balance(["Current Liabilities", "Total Current Liabilities"])
    retained_earnings = fetch_balance(["Retained Earnings"])
    ebit = fetch_financials(["EBIT", "Operating Income"])
    sales = fetch_financials(["Total Revenue", "Sales"])

    share_price = info.get('regularMarketPrice')
    shares_outstanding = info.get('sharesOutstanding')
    market_cap = share_price * shares_outstanding if share_price and shares_outstanding else None

    if not all([total_assets, total_liabilities, market_cap]):
        return None, "Essential data missing for computation."

    working_capital = (current_assets - current_liabilities) if current_assets and current_liabilities else 0
    ratio1 = working_capital / total_assets
    ratio2 = retained_earnings / total_assets
    ratio3 = ebit / total_assets
    ratio4 = market_cap / total_liabilities
    ratio5 = sales / total_assets

    z_score = 1.2 * ratio1 + 1.4 * ratio2 + 3.3 * ratio3 + 0.6 * ratio4 + ratio5

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

def extract_features(tickers):
    records = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            dy = info.get('dividendYield', np.nan)
            growth = info.get('earningsGrowth', np.nan)
            price = info.get('regularMarketPrice', np.nan)
            beta = info.get('beta', np.nan)
            expected_return = (dy or 0) + (growth or 0)
        except Exception:
            dy, growth, price, beta, expected_return = np.nan, np.nan, np.nan, np.nan, np.nan
        records.append([ticker, dy, price, beta, expected_return])

    return pd.DataFrame(records, columns=['Ticker', 'Dividend Yield', 'Price', 'Stability', 'Expected Return'])

def remove_outliers(df, columns):
    z_scores = np.abs(stats.zscore(df[columns].dropna()))
    df_clean = df[(z_scores < 3).all(axis=1)]
    return df_clean

def perform_clustering(df):
    df_clean = df.dropna(subset=['Dividend Yield', 'Expected Return', 'Stability'])
    df_clean = remove_outliers(df_clean, ['Dividend Yield', 'Expected Return', 'Stability'])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df_clean[['Dividend Yield', 'Expected Return', 'Stability']])
    model = KMeans(n_clusters=3, random_state=42)
    df_clean['Cluster'] = model.fit_predict(features_scaled)
    return model, df_clean

def recommend_stocks(df, budget, model=None, preferences=None, min_price_per_stock=20, max_price_per_stock=500):
    df_clean = df.dropna(subset=['Dividend Yield', 'Expected Return', 'Stability'])
    df_clean = remove_outliers(df_clean, ['Dividend Yield', 'Expected Return', 'Stability'])

    if preferences:
        priority = preferences.get('priority')
        if priority == 'Dividend Yield':
            df_clean = df_clean.sort_values('Dividend Yield', ascending=False)
        elif priority == 'Expected Return':
            df_clean = df_clean.sort_values('Expected Return', ascending=False)
        elif priority == 'Stability':
            df_clean = df_clean.sort_values('Stability', ascending=False)

    if model:
        features = df_clean[['Dividend Yield', 'Expected Return', 'Stability']]
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        df_clean['Cluster'] = model.predict(features_scaled)
        best_cluster = df_clean['Cluster'].mode()[0]
        df_clean = df_clean[df_clean['Cluster'] == best_cluster]

    df_clean = df_clean[(df_clean['Price'] >= min_price_per_stock) & 
                        (df_clean['Price'] <= max_price_per_stock)]

    selected = df_clean.head(5)
    allocation = budget / len(selected) if len(selected) > 0 else 0
    selected['Allocation'] = allocation

    return selected

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    df = pd.read_html(str(table))[0]
    return df['Symbol'].tolist()

def display_sector_density():
    st.title("🌌 Sector Density Explorer")

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

    # --- Dropdown: Sector Exploration ---
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

    # --- NEW: Stock Proximity Explorer ---
    st.markdown("### 🔍 Stock Proximity Explorer")
    query = st.text_input("Enter a Ticker (e.g., TICK123)")

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
            title=f"🧭 Position of {query} Among Competitors"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif query:
        st.warning("❌ Ticker not found. Try examples like TICK0, TICK42, TICK1987")

# ============================================
# Streamlit Main App
# ============================================

def main():
    st.title("Financial Dashboard")

    page = st.sidebar.radio(
        "Navigation", 
        ["Dividend Dashboard", "Altman Z-Score", "Investing Analysis", "Sector Density Explorer", "Explain Backend"]
    )

    if page == "Dividend Dashboard":
        ticker = st.text_input("Enter Ticker", "AAPL")
        if st.button("Show Dividend Info"):
            display_dividend_dashboard(ticker)

    elif page == "Altman Z-Score":
        ticker = st.text_input("Enter Ticker for Z-Score", "AAPL", key="zscore")
        if st.button("Compute Altman Z-Score"):
            z_score, classification = compute_altman_z(ticker)
            if z_score:
                st.success(f"Altman Z-Score: {z_score:.2f}")
                st.info(f"Classification: {classification}")
            else:
                st.error(f"Error: {classification}")

    elif page == "Investing Analysis":
        # your previous Investing Analysis page code
        st.subheader("Input preferences below for personalized investment analysis:")
        # [the rest of your investing code here]

    elif page == "Sector Density Explorer":
        display_sector_density()

    elif page == "Explain Backend":
        explain_backend()

if __name__ == "__main__":
    main()
