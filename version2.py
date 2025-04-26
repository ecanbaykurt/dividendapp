import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

############################################
# DIVIDEND DASHBOARD FUNCTIONS
############################################

def display_dividend_dashboard(ticker: str):
    t_obj = yf.Ticker(ticker)
    info = t_obj.info

    st.subheader("Company Overview")
    st.write(info.get('longBusinessSummary', "No overview available."))

    st.subheader("Dividend History (Last 10 Entries)")
    dividends = t_obj.dividends
    if dividends.empty:
        st.write("No dividend data available for this ticker.")
    else:
        data_to_plot = dividends.tail(10) if len(dividends) > 10 else dividends
        st.write(data_to_plot)
        fig_div, ax_div = plt.subplots(figsize=(10, 4))
        ax_div.bar(data_to_plot.index, data_to_plot)
        ax_div.set_title("Dividend History (Last 10 Entries)")
        ax_div.set_xlabel("Date")
        ax_div.set_ylabel("Dividend ($)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_div)

    st.subheader("Price History (Last 1 Year)")
    price_history = t_obj.history(period="1y")
    if price_history.empty:
        st.write("No price data available for this ticker.")
    else:
        st.write(price_history[['Close']].head())
        fig_price, ax_price = plt.subplots(figsize=(10, 4))
        ax_price.plot(price_history.index, price_history['Close'], label="Closing Price")
        ax_price.set_title("Price History (Last 1 Year)")
        ax_price.set_xlabel("Date")
        ax_price.set_ylabel("Price ($)")
        ax_price.legend()
        st.pyplot(fig_price)

    st.subheader("Key Financial Metrics")
    trailing_eps = info.get('trailingEps', None)
    dividend_rate = info.get('dividendRate', None)
    dividend_yield = info.get('dividendYield', None)
    dividend_payout_ratio = (dividend_rate / trailing_eps) if trailing_eps and dividend_rate else None

    st.write("Trailing EPS:", trailing_eps or "N/A")
    st.write("Dividend Rate:", dividend_rate or "N/A")
    st.write("Dividend Yield:", dividend_yield or "N/A")
    if dividend_payout_ratio:
        st.write("Dividend Payout Ratio:", round(dividend_payout_ratio, 2))
    else:
        st.write("Dividend payout ratio could not be calculated.")

############################################
# ALTMAN Z-SCORE FUNCTIONS
############################################

def compute_altman_z(ticker: str):
    t_obj = yf.Ticker(ticker)
    bs = t_obj.balance_sheet
    fs = t_obj.financials
    info = t_obj.info

    # Fetch necessary balance sheet and financial data
    def get_bs_value(keys):
        for key in keys:
            for bs_key in bs.index:
                if bs_key.strip().lower() == key.strip().lower():
                    return bs.loc[bs_key][bs.columns[0]]
        return None

    def get_fs_value(keys):
        for key in keys:
            for fs_key in fs.index:
                if fs_key.strip().lower() == key.strip().lower():
                    return fs.loc[fs_key][fs.columns[0]]
        return None

    # Main data fetching
    total_assets = get_bs_value(["Total Assets"])
    total_liabilities = get_bs_value([
        "Total Liabilities",
        "Total Liabilities Net Minority Interest"
    ])
    current_assets = get_bs_value(["Current Assets", "Total Current Assets"])
    current_liabilities = get_bs_value(["Current Liabilities", "Total Current Liabilities"])
    retained_earnings = get_bs_value(["Retained Earnings"])
    ebit = get_fs_value(["EBIT", "Operating Income"])
    sales = get_fs_value(["Total Revenue", "Sales"])

    # Market value calculation
    share_price = info.get('regularMarketPrice', None)
    shares_outstanding = info.get('sharesOutstanding', None)
    market_value_of_equity = share_price * shares_outstanding if share_price and shares_outstanding else None

    # If essential data is missing
    if not all([total_assets, total_liabilities, market_value_of_equity]):
        return None, "Essential data missing for ticker."

    # Z-Score calculations
    working_capital = current_assets - current_liabilities if current_assets and current_liabilities else 0
    ratio1 = working_capital / total_assets
    ratio2 = retained_earnings / total_assets
    ratio3 = ebit / total_assets
    ratio4 = market_value_of_equity / total_liabilities
    ratio5 = sales / total_assets

    z_score = 1.2 * ratio1 + 1.4 * ratio2 + 3.3 * ratio3 + 0.6 * ratio4 + ratio5
    classification = "Safe Zone" if z_score > 2.99 else "Grey Zone" if z_score >= 1.81 else "Distressed Zone"
    
    return z_score, classification

############################################
# INVESTING ANALYSIS FUNCTIONS
############################################

def extract_features(tickers):
    data = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            dy = info.get('dividendYield', np.nan)
            er = info.get('regularMarketPrice', np.nan)
            stl = info.get('beta', np.nan)
        except Exception:
            dy, er, stl = np.nan, np.nan, np.nan
        data.append([t, dy, er, stl])
    return pd.DataFrame(data, columns=['Ticker', 'Dividend Yield', 'Expected Return', 'Stability'])

def perform_clustering(df):
    # Handle missing values before clustering
    df = df.dropna(subset=['Dividend Yield', 'Expected Return', 'Stability'])  # Drop rows with missing data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[['Dividend Yield', 'Expected Return', 'Stability']].fillna(0))  # Fill any remaining NaNs with 0
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df_scaled)
    
    return kmeans, df

def recommend_stocks(df, budget, cluster_model=None, user_preferences=None):
    # Handle missing data
    df = df.dropna(subset=['Dividend Yield', 'Expected Return', 'Stability'])  # Drop rows with missing values
    
    # Apply user preferences (e.g., prioritize dividend yield, stability, or expected return)
    if user_preferences:
        if user_preferences.get('priority') == 'Dividend Yield':
            df = df.sort_values('Dividend Yield', ascending=False)
        elif user_preferences.get('priority') == 'Expected Return':
            df = df.sort_values('Expected Return', ascending=False)
        elif user_preferences.get('priority') == 'Stability':
            df = df.sort_values('Stability', ascending=False)
    
    # If clustering model is passed, use it to cluster the stocks first
    if cluster_model:
        df['Cluster'] = cluster_model.predict(df[['Dividend Yield', 'Expected Return', 'Stability']].fillna(0))
        # Filter stocks from the best cluster for the user
        best_cluster = df['Cluster'].mode()[0]
        df = df[df['Cluster'] == best_cluster]
    
    # Sorting by dividend yield
    top = df.head(5)  # Select top 5 based on user preferences
    alloc = budget / len(top)  # Even allocation of budget
    top['Allocation'] = alloc
    return top

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    df = pd.read_html(str(table))[0]
    return df['Symbol'].tolist()

############################################
# STREAMLIT APP
############################################

def main():
    st.title("Personalized Financial Dashboard")

    page = st.sidebar.radio("Select Page", [
        "Dividend Dashboard",
        "Altman Z-Score",
        "Investing Analysis",
        "Explain Backend"
    ])

    # Personalization options
    user_preferences = {
        'priority': st.sidebar.selectbox("Choose your investment priority", ['Dividend Yield', 'Expected Return', 'Stability'])
    }

    if page == "Dividend Dashboard":
        ticker = st.text_input("Ticker", "AAPL")
        if st.button("Show Dividend Data"):
            display_dividend_dashboard(ticker)

    elif page == "Altman Z-Score":
        ticker = st.text_input("Ticker", "AAPL", key="alt")
        if st.button("Calculate Z-Score"):
            result = compute_altman_z(ticker)
            if result[0]:
                z_score, classification = result
                st.success(f"Altman Z-Score: {z_score:.2f}")
                st.info(f"Classification: {classification}")
            else:
                st.error("Error: " + result[1])

    elif page == "Investing Analysis":
        budget = st.number_input("Investment Amount ($)", min_value=0)
        if st.button("Recommend Top Dividend Stocks"):
            tickers = get_sp500_tickers()
            df = extract_features(tickers)
            kmeans, clustered_stocks = perform_clustering(df)  # Perform clustering
            recommended_stocks = recommend_stocks(clustered_stocks, budget, cluster_model=kmeans, user_preferences=user_preferences)
            st.write("Recommended Stocks for Investment:")
            st.write(recommended_stocks)

    elif page == "Explain Backend":
        explain_backend()

if __name__ == "__main__":
    main()
