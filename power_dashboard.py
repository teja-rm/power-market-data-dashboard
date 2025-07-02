import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sqlite3
import io

# Configure page
st.set_page_config(
    page_title="Power Market Analytics Dashboard",
    page_icon="⚡",
    layout="wide"
)

class PowerMarketData:
    def __init__(self):
        self.db_name = "power_market.db"
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                region TEXT,
                price REAL,
                demand REAL,
                renewable_percentage REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    def generate_sample_data(self, hours=168):  # 1 week of data
        """Generate realistic sample power market data"""
        np.random.seed(42)
        regions = ['DK1', 'DK2', 'SE1', 'SE2', 'NO1', 'NO2']
        data = []
        
        base_time = datetime.now() - timedelta(hours=hours)
        
        for region in regions:
            base_price = np.random.uniform(30, 80)  # EUR/MWh
            base_demand = np.random.uniform(1000, 3000)  # MW
            
            for i in range(hours):
                timestamp = base_time + timedelta(hours=i)
                
                # Add daily and weekly patterns
                hour_of_day = timestamp.hour
                day_of_week = timestamp.weekday()
                
                # Price variations
                daily_factor = 1 + 0.3 * np.sin(2 * np.pi * hour_of_day / 24)
                weekly_factor = 1 + 0.1 * np.sin(2 * np.pi * day_of_week / 7)
                noise = np.random.normal(0, 0.1)
                
                price = base_price * daily_factor * weekly_factor * (1 + noise)
                price = max(0, price)  # Ensure non-negative prices
                
                # Demand variations (inverse correlation with price)
                demand = base_demand * (2 - daily_factor) * weekly_factor * (1 + noise * 0.5)
                demand = max(0, demand)
                
                # Renewable percentage (higher during day, varies by region)
                renewable_base = 0.4 if region.startswith('DK') else 0.6  # Denmark vs Nordic
                renewable_pct = renewable_base + 0.2 * np.sin(2 * np.pi * hour_of_day / 24) + np.random.normal(0, 0.05)
                renewable_pct = np.clip(renewable_pct, 0, 1)
                
                data.append({
                    'timestamp': timestamp,
                    'region': region,
                    'price': round(price, 2),
                    'demand': round(demand, 1),
                    'renewable_percentage': round(renewable_pct, 3)
                })
        
        return pd.DataFrame(data)
    
    def load_data(self):
        """Load data from database or generate if empty"""
        conn = sqlite3.connect(self.db_name)
        df = pd.read_sql_query("SELECT * FROM market_data", conn)
        conn.close()
        
        if df.empty:
            df = self.generate_sample_data()
            self.save_data(df)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def save_data(self, df):
        """Save data to database"""
        conn = sqlite3.connect(self.db_name)
        df.to_sql('market_data', conn, if_exists='replace', index=False)
        conn.close()

def main():
    st.title("⚡ Power Market Analytics Dashboard")
    st.markdown("Real-time electricity market data visualization and analysis")
    
    # Initialize data handler
    data_handler = PowerMarketData()

    # Sidebar controls
    st.sidebar.header("Dashboard Controls")

    # Refresh data button
    if st.sidebar.button("Refresh Data"):
        df_new = data_handler.generate_sample_data()
        data_handler.save_data(df_new)
        st.sidebar.success("Data refreshed!")

    # Load data
    with st.spinner("Loading market data..."):
        df = data_handler.load_data()

    # Region selection
    regions = df['region'].unique()
    selected_regions = st.sidebar.multiselect(
        "Select Regions",
        regions,
        default=regions[:3]
    )

    # Time range selection
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()

    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Time slider for hour selection
    min_hour = int(df['timestamp'].dt.hour.min())
    max_hour = int(df['timestamp'].dt.hour.max())
    #selected_hour = st.sidebar.slider("Select Hour of Day", min_hour, max_hour, (min_hour, max_hour))
    selected_hour_range = st.sidebar.slider("Select Hour Range", min_hour, max_hour, (min_hour, max_hour))
    selected_hour_min, selected_hour_max = selected_hour_range

    # Filter data
    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
        #mask &= (df['timestamp'].dt.hour >= selected_hour) & (df['timestamp'].dt.hour <= selected_hour)
        mask &= (df['timestamp'].dt.hour >= selected_hour_min) & (df['timestamp'].dt.hour <= selected_hour_max)
        filtered_df = df[mask & df['region'].isin(selected_regions)]
    else:
        filtered_df = df[df['region'].isin(selected_regions)]

    if filtered_df.empty:
        st.warning("No data available for selected filters.")
        return

    # Main dashboard layout
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_price = filtered_df['price'].mean()
        median_price = filtered_df['price'].median()
        st.metric("Average Price", f"{avg_price:.2f} EUR/MWh")
        st.write(f"Median Price: {median_price:.2f} EUR/MWh")

    with col2:
        total_demand = filtered_df['demand'].sum()
        min_demand = filtered_df['demand'].min()
        max_demand = filtered_df['demand'].max()
        st.metric("Total Demand", f"{total_demand:,.0f} MW")
        st.write(f"Min Demand: {min_demand:.1f} MW, Max Demand: {max_demand:.1f} MW")

    with col3:
        avg_renewable = filtered_df['renewable_percentage'].mean() * 100
        st.metric("Avg Renewable %", f"{avg_renewable:.1f}%")

    with col4:
        price_volatility = filtered_df['price'].std()
        st.metric("Price Volatility", f"{price_volatility:.2f}")

    # Price trends chart
    st.subheader("Price Trends by Region")
    fig_price = px.line(
        filtered_df,
        x='timestamp',
        y='price',
        color='region',
        title="Electricity Prices Over Time",
        labels={'price': 'Price (EUR/MWh)', 'timestamp': 'Time'}
    )
    fig_price.update_layout(height=400)
    st.plotly_chart(fig_price, use_container_width=True)

    # Demand vs Price correlation
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Price vs Demand Correlation")
        fig_scatter = px.scatter(
            filtered_df,
            x='demand',
            y='price',
            color='region',
            title="Price vs Demand by Region",
            labels={'demand': 'Demand (MW)', 'price': 'Price (EUR/MWh)'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        st.subheader("Renewable Energy Share")
        fig_renewable = px.box(
            filtered_df,
            x='region',
            y='renewable_percentage',
            title="Renewable Energy Percentage by Region"
        )
        fig_renewable.update_yaxes(tickformat='.0%')
        st.plotly_chart(fig_renewable, use_container_width=True)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr = filtered_df[['price', 'demand', 'renewable_percentage']].corr()
    fig_corr = px.imshow(corr, text_auto=True, title="Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)

    # Price alerts
    st.subheader("Price Alerts")
    alert_threshold = st.slider("Price Alert Threshold (EUR/MWh)", 0, 200, 100)

    high_price_alerts = filtered_df[filtered_df['price'] > alert_threshold]

    if not high_price_alerts.empty:
        st.warning(f"⚠️ {len(high_price_alerts)} price alerts above {alert_threshold} EUR/MWh")
        
        alert_summary = high_price_alerts.groupby('region').agg({
            'price': ['max', 'mean', 'count'],
            'timestamp': ['min', 'max']
        }).round(2)
        
        st.dataframe(alert_summary)
    else:
        st.success("✅ No price alerts for selected threshold")

    # Data export
    st.subheader("Data Export")
    if st.button("Download Filtered Data as CSV"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"power_market_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    if st.button("Download Filtered Data as Excel"):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            filtered_df.to_excel(writer, index=False, sheet_name='MarketData')
            writer.close()
        processed_data = output.getvalue()
        st.download_button(
            label="Download Excel",
            data=processed_data,
            file_name=f"power_market_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # Show raw data table
    st.subheader("Raw Data Table")
    st.dataframe(filtered_df.reset_index(drop=True))

if __name__ == "__main__":
    main()
