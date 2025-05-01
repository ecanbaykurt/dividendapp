# ============================================
# Imports and Page Config
# ============================================
import streamlit as st
st.set_page_config(page_title="Financial Dashboard — Christine, Omar, Emre (BA870)", layout="wide")

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
from sklearn.datasets import make_blobs
import plotly.graph_objects as go
import plotly.express as px
from sklearn.neighbors import KernelDensity

import requests
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

FMP_API_KEY = "493XsCeiSC8NLp2MQCZyWbwKbHQ85wbE"

def display_dividend_dashboard(ticker: str):
    base_url = "https://financialmodelingprep.com/api/v3"

    # 1. Company Overview
    profile = requests.get(f"{base_url}/profile/{ticker}?apikey={FMP_API_KEY}").json()
    if not profile:
        st.error("❌ Ticker not found or API issue.")
        return
    info = profile[0]

    st.subheader("Company Overview")
    st.write(info.get("description", "No overview available."))

    # 2. Dividend History
    divs = requests.get(f"{base_url}/historical-price-full/stock_dividend/{ticker}?apikey={FMP_API_KEY}").json()
    if "historical" in divs and divs["historical"]:
        df_div = pd.DataFrame(divs["historical"]).head(10)
        df_div["date"] = pd.to_datetime(df_div["date"])
        st.subheader("Dividend History (Last 10 Entries)")
        st.dataframe(df_div[["date", "dividend"]])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(df_div["date"], df_div["dividend"])
        ax.set_title("Dividend History")
        ax.set_xlabel("Date")
        ax.set_ylabel("Dividend ($)")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.subheader("Dividend History")
        st.info("No dividend history available.")

    # 3. Price History
    prices = requests.get(f"{base_url}/historical-price-full/{ticker}?timeseries=365&apikey={FMP_API_KEY}").json()
    if "historical" in prices and prices["historical"]:
        df_price = pd.DataFrame(prices["historical"])
        df_price["date"] = pd.to_datetime(df_price["date"])
        df_price.set_index("date", inplace=True)
        df_price.sort_index(inplace=True)

        st.subheader("Price History (Last 1 Year)")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_price.index, df_price["close"], label="Close Price")
        ax.set_title("Price History (Last 1 Year)")
        ax.set_ylabel("Price ($)")
        ax.set_xlabel("Date")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("No price history found.")

    # 4. Key Financial Metrics
    st.subheader("Key Financial Metrics")
    eps = info.get("eps")
    dividend = info.get("lastDiv")
    price = info.get("price")
    dividend_yield = dividend / price if price else None
    payout_ratio = dividend / eps if eps else None

    st.write("Trailing EPS:", eps or "N/A")
    st.write("Dividend Rate:", dividend or "N/A")
    st.write("Dividend Yield:", f"{dividend_yield*100:.2f}%" if dividend_yield else "N/A")
    st.write("Payout Ratio:", round(payout_ratio, 2) if payout_ratio else "N/A")

# ============================================
# Altman Z-Score Functions
# ============================================
def compute_altman_z(ticker: str):
    base_url = "https://financialmodelingprep.com/api/v3"

    # Pull data from FMP endpoints
    bs_url = f"{base_url}/balance-sheet-statement/{ticker}?limit=1&apikey={FMP_API_KEY}"
    is_url = f"{base_url}/income-statement/{ticker}?limit=1&apikey={FMP_API_KEY}"
    profile_url = f"{base_url}/profile/{ticker}?apikey={FMP_API_KEY}"

    try:
        bs_data = requests.get(bs_url).json()[0]
        is_data = requests.get(is_url).json()[0]
        profile_data = requests.get(profile_url).json()[0]
    except Exception as e:
        return None, f"API error: {e}"

    try:
        # Financial Metrics
        total_assets = bs_data["totalAssets"]
        total_liabilities = bs_data["totalLiabilities"]
        current_assets = bs_data["totalCurrentAssets"]
        current_liabilities = bs_data["totalCurrentLiabilities"]
        retained_earnings = bs_data["retainedEarnings"]
        ebit = is_data["ebit"]
        revenue = is_data["revenue"]

        share_price = profile_data["price"]
        market_cap = profile_data["mktCap"]

        # Altman Z-score ratios
        working_capital = current_assets - current_liabilities
        ratio1 = working_capital / total_assets
        ratio2 = retained_earnings / total_assets
        ratio3 = ebit / total_assets
        ratio4 = market_cap / total_liabilities
        ratio5 = revenue / total_assets

        z_score = 1.2 * ratio1 + 1.4 * ratio2 + 3.3 * ratio3 + 0.6 * ratio4 + ratio5

        if z_score > 2.99:
            classification = "Safe Zone"
        elif z_score >= 1.81:
            classification = "Grey Zone"
        else:
            classification = "Distressed Zone"

        return z_score, classification
    except Exception as e:
        return None, f"Computation error: {e}"

# ============================================
# Investing Analysis Using Local CSV
# ============================================

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import streamlit as st

# ============================================
# Load Data from CSV (No API)
# ============================================
@st.cache_data(show_spinner=False)
def extract_features():
    try:
        df = pd.read_csv("sp500_profile_data.csv")
        required_cols = ['Ticker', 'Dividend Yield', 'Price', 'Stability', 'Expected Return']
        if not all(col in df.columns for col in required_cols):
            st.error("❌ CSV missing required columns.")
            return pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"❌ Failed to load CSV: {e}")
        return pd.DataFrame()

# ============================================
# Remove Outliers
# ============================================
def remove_outliers(df, columns):
    try:
        z_scores = np.abs(stats.zscore(df[columns].dropna()))
        return df[(z_scores < 3).all(axis=1)]
    except Exception:
        return df

# ============================================
# Clustering Function
# ============================================
def perform_clustering(df):
    df_clean = df.dropna(subset=['Dividend Yield', 'Expected Return', 'Stability'])
    df_clean = remove_outliers(df_clean, ['Dividend Yield', 'Expected Return', 'Stability'])

    if df_clean.empty:
        raise ValueError("No valid data left after cleaning.")

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df_clean[['Dividend Yield', 'Expected Return', 'Stability']])

    model = KMeans(n_clusters=3, random_state=42)
    df_clean['Cluster'] = model.fit_predict(features_scaled)

    return model, df_clean

# ============================================
# Recommender Function
# ============================================
def recommend_stocks(df, budget, model=None, preferences=None, min_price_per_stock=20, max_price_per_stock=500):
    df_clean = df.dropna(subset=['Dividend Yield', 'Expected Return', 'Stability'])
    df_clean = remove_outliers(df_clean, ['Dividend Yield', 'Expected Return', 'Stability'])

    if preferences:
        priority = preferences.get('priority')
        if priority in df_clean.columns:
            df_clean = df_clean.sort_values(priority, ascending=False)

    if model:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(df_clean[['Dividend Yield', 'Expected Return', 'Stability']])
        df_clean['Cluster'] = model.predict(features_scaled)
        best_cluster = df_clean['Cluster'].mode()[0]
        df_clean = df_clean[df_clean['Cluster'] == best_cluster]

    df_clean = df_clean[(df_clean['Price'] >= min_price_per_stock) & (df_clean['Price'] <= max_price_per_stock)]

    if df_clean.empty:
        return pd.DataFrame()

    selected = df_clean.head(5)
    selected['Allocation'] = budget / len(selected)
    return selected

#Sector Competitor Explorer

def sector_competitor_explorer():
    st.title("📈 Sector Competitor Explorer (Custom Dataset)")

    try:
        trimmed_df = pd.read_csv("your_cleaned_trimmed_df.csv")
    except Exception as e:
        st.error("❌ Data file could not be loaded. Please make sure 'your_cleaned_trimmed_df.csv' exists.")
        return

    ticker_input = st.text_input("Enter a Ticker to Find Sector Competitors", "AAPL").upper()

    if st.button("Find Competitors"):
        if ticker_input in trimmed_df['ticker'].values:
            sector = trimmed_df.loc[trimmed_df['ticker'] == ticker_input, 'sector'].values[0]
            competitors = trimmed_df[trimmed_df['sector'] == sector]
            st.success(f"Sector: {sector}")
            st.write(f"Found {len(competitors)} companies in this sector:")
            st.dataframe(competitors[['ticker', 'sector', 'profitability_ratio']].reset_index(drop=True))
        else:
            st.error("❌ Ticker not found in the dataset.")

# --- New Function: Hidden Competitor Neural Map ---
def hidden_competitor_neural_map():
    st.title("🧠 Hidden Competitor Neural Map")

    try:
        trimmed_df = pd.read_csv("your_cleaned_trimmed_df.csv")
        umap_embeddings_3d = np.load("your_umap_embeddings.npy")
    except Exception as e:
        st.error("❌ Data files not found. Please check your CSV and NPY files.")
        st.stop()

    plot_df_3d = pd.DataFrame({
        'x': umap_embeddings_3d[:, 0],
        'y': umap_embeddings_3d[:, 1],
        'z': umap_embeddings_3d[:, 2],
        'ticker': trimmed_df['ticker'],
        'sector': trimmed_df['sector'],
        'cluster': trimmed_df['hidden_competitor_cluster']
    })

    view_mode = st.radio("Choose View Mode", ["🔥 Sector Density Heatmap", "🌐 All Industry Cluster Map"])

    if view_mode == "🔥 Sector Density Heatmap":
        sectors = sorted(plot_df_3d['sector'].unique())
        selected_sector = st.sidebar.selectbox("Select Sector", sectors)
        sector_data = plot_df_3d[plot_df_3d['sector'] == selected_sector]
        if len(sector_data) < 10:
            st.warning("Not enough data points for density.")
            return
        xyz = np.vstack([sector_data['x'], sector_data['y'], sector_data['z']]).T
        kde = KernelDensity(bandwidth=0.5, kernel='gaussian').fit(xyz)
        density = np.exp(kde.score_samples(xyz))
        sector_data['density'] = density

        fig = px.scatter_3d(
            sector_data, x='x', y='y', z='z', color='density',
            color_continuous_scale='Hot', text='ticker',
            hover_data=['ticker', 'cluster', 'density'],
            title=f"🔥 Density Heatmap - {selected_sector} Sector"
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        cluster_ids = plot_df_3d['cluster'].unique()
        colors = px.colors.qualitative.Plotly
        color_map = {cid: colors[i % len(colors)] for i, cid in enumerate(cluster_ids)}

        fig = go.Figure()
        for cluster_id in cluster_ids:
            cluster_data = plot_df_3d[plot_df_3d['cluster'] == cluster_id]
            fig.add_trace(go.Scatter3d(
                x=cluster_data['x'], y=cluster_data['y'], z=cluster_data['z'],
                mode='markers',
                marker=dict(size=5, color=color_map[cluster_id], opacity=0.8),
                text=cluster_data['ticker'] + " (" + cluster_data['sector'] + ")",
                hoverinfo='text'
            ))

        fig.update_layout(
            title="🌐 Hidden Competitor Map — All Industries",
            scene=dict(xaxis_title='UMAP-1', yaxis_title='UMAP-2', zaxis_title='UMAP-3'),
            width=1100, height=900, showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# Explain Backend Functions
# ============================================
def explain_backend():
    st.title("🛠️ Backend Code Explanation")

    st.markdown("""
This section explains the backend logic for each component of the Financial Dashboard app.
The app uses data from Yahoo Finance and custom datasets to deliver analytical insights and recommendations.
    """)

    st.markdown("## 📦 Imports & Setup")
    st.code("""
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
    """)
    st.write("These libraries are used for building the user interface, retrieving financial data, performing clustering, and rendering visualizations.")

    # === Dividend Dashboard ===
    with st.expander("💰 Dividend Dashboard Code"):
        st.write("Fetches and displays dividend history, company overview, and financial ratios for a given ticker using the yfinance API.")
        st.code("def display_dividend_dashboard(ticker: str):")
        st.markdown("""
- Uses `yf.Ticker().dividends` to get dividend history  
- Uses `matplotlib` to plot recent dividend bars  
- Calculates dividend payout ratio using EPS and dividend rate  
        """)

    # === Altman Z-Score ===
    with st.expander("📉 Altman Z-Score Code"):
        st.write("Retrieves key financial statement values and applies the Altman Z-Score formula to assess bankruptcy risk.")
        st.code("def compute_altman_z(ticker: str):")
        st.markdown("""
- Parses balance sheet and income statement  
- Calculates five key ratios  
- Combines ratios using Altman Z-Score formula  
- Classifies result into Safe, Grey, or Distressed zones  
        """)

    # === Investing Analysis ===
    with st.expander("📈 Investing Analysis Code"):
        st.write("Fetches S&P 500 tickers, retrieves financial metrics, clusters them, and recommends stocks based on user preferences.")
        st.markdown("""
- Uses `extract_features()` to collect Dividend Yield, Growth, Beta  
- `perform_clustering()` groups stocks into 3 clusters  
- `recommend_stocks()` filters and allocates portfolio based on user budget and strategy  
- Calculates expected annual dividend income  
        """)

    # === Sector Competitor Explorer ===
    with st.expander("🏷️ Sector Competitor Explorer Code"):
        st.write("Finds companies in the same sector based on the user's selected ticker symbol from a custom CSV dataset.")
        st.code("""
def sector_competitor_explorer():
    trimmed_df = pd.read_csv("your_cleaned_trimmed_df.csv")
    ticker_input = st.text_input("Enter a Ticker", "AAPL").upper()

    if st.button("Find Competitors"):
        if ticker_input in trimmed_df['ticker'].values:
            sector = trimmed_df.loc[trimmed_df['ticker'] == ticker_input, 'sector'].values[0]
            competitors = trimmed_df[trimmed_df['sector'] == sector]
            st.dataframe(competitors[['ticker', 'sector', 'profitability_ratio']])
        else:
            st.error("Ticker not found in the dataset.")
        """)

    # === Hidden Competitor Neural Map ===
    with st.expander("🌌 Hidden Competitor Neural Map Code"):
        st.write("Visualizes companies in a 3D space using UMAP embeddings based on business descriptions and clusters.")
        st.markdown("""
- Loads embeddings from `.npy` file and joins with company info  
- Displays a 3D scatter plot using Plotly  
- Offers two views: full industry map or sector-specific density heatmap  
        """)

# ============================================
# Streamlit Main App (CSV-based)
# ============================================
def main():
    st.title("🏦 Financial Dashboard — Christine, Omar, Emre (BA870)")

    page = st.sidebar.radio(
        "Navigation",
        ["Dividend Dashboard", "Altman Z-Score", "Investing Analysis", "Sector Competitor Explorer", "Hidden Competitor Neural Map", "Explain Backend"]
    )

    if page == "Dividend Dashboard":
        ticker = st.text_input("Enter Ticker", "AAPL")
        if st.button("Show Dividend Info"):
            display_dividend_dashboard(ticker)

    elif page == "Altman Z-Score":
        ticker = st.text_input("Enter Ticker for Z-Score", "AAPL")
        if st.button("Compute Altman Z-Score"):
            z_score, classification = compute_altman_z(ticker)
            if z_score:
                st.success(f"Altman Z-Score: {z_score:.2f}")
                st.info(f"Classification: {classification}")
            else:
                st.error(f"Error: {classification}")

    elif page == "Investing Analysis":
        st.subheader("📈 Personalized Investment Recommendation")
        budget = st.number_input("Enter Investment Budget ($)", min_value=1000, value=2000)
        investment_priority = st.selectbox("Select Investment Priority", ['Dividend Yield', 'Expected Return', 'Stability'])
        min_price = st.number_input("Minimum Stock Price ($)", min_value=0, value=20)
        max_price = st.number_input("Maximum Stock Price ($)", min_value=0, value=500)

        if st.button("Get Stock Recommendations"):
            with st.spinner("Fetching and analyzing CSV data..."):
                df_features = extract_features()  # ✅ Load from local CSV
                if df_features.empty:
                    st.error("⚠️ No data available. Please check your CSV file.")
                    return

                model, clustered = perform_clustering(df_features)

                preferences = {'priority': investment_priority}
                recommended_stocks = recommend_stocks(clustered, budget, model, preferences, min_price, max_price)

            st.subheader("💡 Top Recommended Stocks")
            st.write(recommended_stocks)

            # Display high-yield options if available
            high_yield_df = df_features[df_features['Dividend Yield'] >= 0.04]
            if not high_yield_df.empty:
                st.subheader("🔥 High-Yield Stocks (≥ 4%) Available in Dataset")
                st.dataframe(high_yield_df[['Ticker', 'Dividend Yield', 'Price', 'Expected Return']].sort_values(by='Dividend Yield', ascending=False))
            else:
                st.info("ℹ️ No stocks in the dataset offer a dividend yield ≥ 4%.")

            st.write(f"📊 **Max Dividend Yield in Dataset:** `{df_features['Dividend Yield'].max():.2%}`")

            # Dividend income estimation
            total_dividend_yield = recommended_stocks['Dividend Yield'].mean()
            expected_annual_income = budget * total_dividend_yield if not np.isnan(total_dividend_yield) else 0

            st.subheader("🎯 Dividend Income Goal")
            st.metric(label="Expected Annual Dividend Income", value=f"${expected_annual_income:.2f}")

            # Strategy
            st.subheader("📘 Strategy Recommendation")

            yield_rate = expected_annual_income / budget if budget else 0
            st.markdown(f"**Estimated Yield:** `{yield_rate*100:.2f}%`")

            if yield_rate < 0.04:
                st.warning("⚠️ Your current portfolio is underperforming the common 4% dividend yield benchmark.")
                st.markdown("""
                ### 🧠 Top 3 Strategies to Boost Your Dividend Income:
                1. **Refocus on High-Yield Sectors**  
                   Look into sectors like utilities, REITs, and consumer staples that traditionally offer higher yields.

                2. **Use a Dividend Screener**  
                   Filter stocks with a dividend yield above 4%, a payout ratio under 70%, and consistent dividend growth over 5 years.

                3. **Diversify Across Stable Payors**  
                   Include dividend aristocrats—companies with a proven track record of increasing dividends annually for 25+ years.

                These steps help maximize income while balancing risk. Would you like to see a sample high-yield portfolio?
                """)
            else:
                st.success("✅ Your portfolio aligns well with a stable dividend income strategy.")
                st.markdown("""
                ### 📈 Top 3 Reasons Your Strategy Looks Strong:
                1. **Healthy Dividend Yield**  
                   You're exceeding the typical 4% target, signaling efficient use of capital for income.

                2. **Good Balance of Risk and Return**  
                   Your selected stocks likely combine stability with reasonable growth—ideal for long-term compounding.

                3. **Compounding Power**  
                   If reinvested, your dividends can snowball over time—especially when paired with consistent contributions.

                Keep tracking performance and consider rebalancing quarterly for sustained growth.
                """)

    elif page == "Sector Competitor Explorer":
        sector_competitor_explorer()

    elif page == "Hidden Competitor Neural Map":
        hidden_competitor_neural_map()

    elif page == "Explain Backend":
        explain_backend()


# Only run main() when executed directly
if __name__ == "__main__":
    main()


