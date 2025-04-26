# ============================================
# Investing Analysis Functions
# ============================================

def recommend_stocks(df, budget, model=None, preferences=None, min_price_per_stock=20, max_price_per_stock=500):
    df_clean = df.dropna(subset=['Dividend Yield', 'Expected Return', 'Stability'])

    # Remove outliers
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

    df_clean = df_clean[(df_clean['Expected Return'] >= min_price_per_stock) & 
                        (df_clean['Expected Return'] <= max_price_per_stock)]

    # Corrected budget allocation by stock price
    total_price = df_clean['Expected Return'].sum()  # Sum of all selected stock prices
    df_clean['Allocation'] = df_clean['Expected Return'] / total_price * budget

    # Calculate shares to buy and cost of shares
    df_clean['Shares to Buy'] = (df_clean['Allocation'] / df_clean['Expected Return']).astype(int)
    df_clean['Total Cost'] = df_clean['Shares to Buy'] * df_clean['Expected Return']

    return df_clean.head(5)

# ============================================
# Streamlit App
# ============================================

def main():
    st.title("Financial Dashboard")

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
        st.subheader("Input preferences below for personalized investment analysis:")

        budget = st.number_input("Investment Budget ($)", min_value=0)
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

            # Explain clustering
            st.subheader("How Clustering Works")
            st.write("""
            Stocks are grouped into clusters based on similarities in their dividend yield, expected return, and stability.
            We recommend stocks from the 'best' cluster that matches your selected priority.
            """)

            # Visualize clusters in 3D
            st.subheader("Cluster Visualization (3D)")
            fig = plt.figure(figsize=(15, 15))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(clustered['Dividend Yield'], clustered['Expected Return'], clustered['Stability'], 
                       c=clustered['Cluster'], cmap='viridis')

            ax.set_xlabel('Dividend Yield')
            ax.set_ylabel('Expected Return')
            ax.set_zlabel('Stability')
            ax.set_title('Stock Clusters in 3D')

            # Add cluster labels at the center
            for cluster_num in clustered['Cluster'].unique():
                cluster_data = clustered[clustered['Cluster'] == cluster_num]
                center_x = cluster_data['Dividend Yield'].mean()
                center_y = cluster_data['Expected Return'].mean()
                center_z = cluster_data['Stability'].mean()
                ax.text(center_x, center_y, center_z, f'Cluster {cluster_num}', color='black')

            st.pyplot(fig)

            recommendations = recommend_stocks(df_features, budget, model, {'priority': investment_priority}, min_price, max_price)
            st.write(recommendations)
            st.subheader("Recommendations with Shares to Buy and Total Cost")
            st.write(recommendations[['Ticker', 'Dividend Yield', 'Expected Return', 'Shares to Buy', 'Total Cost']])

    elif page == "Explain Backend":
        explain_backend()

if __name__ == "__main__":
    main()
