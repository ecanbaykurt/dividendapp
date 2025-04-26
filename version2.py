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
            price = info.get('regularMarketPrice', np.nan)
            beta = info.get('beta', np.nan)
        except Exception:
            dy, price, beta = np.nan, np.nan, np.nan
        records.append([ticker, dy, price, beta])

    return pd.DataFrame(records, columns=['Ticker', 'Dividend Yield', 'Expected Return', 'Stability'])

def perform_clustering(df):
    df_clean = df.dropna(subset=['Dividend Yield', 'Expected Return', 'Stability'])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df_clean[['Dividend Yield', 'Expected Return', 'Stability']])

    model = KMeans(n_clusters=3, random_state=42)
    df_clean['Cluster'] = model.fit_predict(features_scaled)

    return model, df_clean

def recommend_stocks(df, budget, model=None, preferences=None, min_price_per_stock=20, max_price_per_stock=500):
    df_clean = df.dropna(subset=['Dividend Yield', 'Expected Return', 'Stability'])

    # Filter based on user preferences
    if preferences:
        priority = preferences.get('priority')
        if priority == 'Dividend Yield':
            df_clean = df_clean.sort_values('Dividend Yield', ascending=False)
        elif priority == 'Expected Return':
            df_clean = df_clean.sort_values('Expected Return', ascending=False)
        elif priority == 'Stability':
            df_clean = df_clean.sort_values('Stability', ascending=False)

    # If clustering is used, filter the top cluster
    if model:
        df_clean['Cluster'] = model.predict(df_clean[['Dividend Yield', 'Expected Return', 'Stability']])
        best_cluster = df_clean['Cluster'].mode()[0]
        df_clean = df_clean[df_clean['Cluster'] == best_cluster]

    # Apply a filter based on the available budget and minimum/maximum stock price
    df_clean = df_clean[(df_clean['Expected Return'] >= min_price_per_stock) & 
                        (df_clean['Expected Return'] <= max_price_per_stock)]

    # Select top N stocks for the recommendation
    selected = df_clean.head(5)
    
    # Allocate the budget across selected stocks
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

def plot_clustering(df, model):
    # Plotting the clustering results
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot with different colors for each cluster
    scatter = ax.scatter(
        df['Dividend Yield'], df['Expected Return'], c=df['Cluster'], cmap='viridis', s=100, alpha=0.6
    )
    
    # Add cluster labels to the plot
    for i, row in df.iterrows():
        ax.text(row['Dividend Yield'], row['Expected Return'], str(row['Cluster']), fontsize=9, ha='center')

    ax.set_title("Clustering of Stocks Based on Investment Criteria")
    ax.set_xlabel("Dividend Yield")
    ax.set_ylabel("Expected Return")

    # Add color legend for clusters
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)

    plt.tight_layout()
    st.pyplot(fig)

    # Explain clusters
    st.write("""
        ### What are Clusters?
        Clusters represent groups of stocks that share similar characteristics based on the criteria you've selected (Dividend Yield, Expected Return, Stability). The clustering algorithm (KMeans) divides stocks into groups that are as similar as possible within the group and as different as possible between groups.

        Stocks in the same cluster are expected to behave similarly in terms of the selected features. By using these clusters, we are focusing on recommending stocks from the cluster that best matches your preferences (i.e., whether you prioritize Dividend Yield, Expected Return, or Stability).

        The numbers on the graph represent the cluster each stock belongs to. The goal is to identify the cluster that aligns with your financial goals and then invest accordingly.
    """)

# ============================================
# Streamlit App
# ============================================

def main():
    st.title("📈 Personalized Financial Dashboard")

    page = st.sidebar.radio(
        "Navigation", 
        ["Dividend Dashboard", "Altman Z-Score", "Investing Analysis", "Explain Backend"]
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
        # Personalized Financial Dashboard
        st.subheader("Personalized Financial Dashboard")

        budget = st.number_input("Investment Budget ($)", min_value=0)
        # Investment Priority Dropdown and Minimum and Maximum Stock Prices
        investment_priority = st.selectbox(
            "Select Investment Priority",
            ['Dividend Yield', 'Expected Return', 'Stability']
        )
        min_price = st.number_input("Minimum Stock Price ($)", min_value=0, value=20)
        max_price = st.number_input("Maximum Stock Price ($)", min_value=0, value=500)

        if st.button("Get Stock Recommendations"):
            tickers = get_sp500_tickers()
            df_features = extract_features(tickers)
            model, clustered = perform_clustering(df_features)
            recommendations = recommend_stocks(
                clustered, budget, model=model, preferences={'priority': investment_priority}, 
                min_price_per_stock=min_price, max_price_per_stock=max_price
            )
            st.write("Top Recommended Stocks:")
            st.dataframe(recommendations)

            # Plot the clustering graph and explain
            plot_clustering(clustered, model)

    elif page == "Explain Backend":
        explain_backend()

if __name__ == "__main__":
    main()
