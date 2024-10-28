import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import numpy as np

# Set the Streamlit page configuration
st.set_page_config(layout="wide", page_title="Trading Strategy Backtest Dashboard")
with st.sidebar:
    st.title(" 📊 Trading Strategy Backtest Dashboard")
    st.markdown("<h4 style='font-size: 20px;'>Created by:</h4>", unsafe_allow_html=True)  # Set font size for the "Created by" text
    linkedin_url = "https://www.linkedin.com/in/mkulis/"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: underline; color: inherit; text-decoration-color: blue; font-size: 20px;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">Matthew A. Kulis</a>', unsafe_allow_html=True)  # Set font size for "Matt Kulis"

# File paths for the datasets
data_file_paths = {
    "MSTR": r"C:\Users\User\Desktop\pyton\Streamlit Dashboard\MSTR_2019_to_Present_(10-24-2024).xlsx",
    "MSFT": r"C:\Users\User\Desktop\pyton\Streamlit Dashboard\MSFT_OHLCV_1min_2024-08-01_to_2024-10-25.csv",
    "NVDA": r"C:\Users\User\Desktop\pyton\Streamlit Dashboard\NVDA_OHLCV_1min_2023-01-01_to_2024-10-25.csv",
    # feel free to add additional Ticker data here just follow the same format and don't forget to use a comma after
}

def load_and_prepare_data(file_path):
    """Load and prepare the data from Excel or CSV file."""
    try:
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            return pd.DataFrame()

        data = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['9ema'] = data['close'].ewm(span=9, adjust=False).mean()
        data['is_red'] = data['open'] > data['close']
        # Calculate the average price for execution
        data['execution_price'] = (data['open'] + data['high'] + data['low'] + data['close']) / 4
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def get_date_range():
    """Get date range selection from sidebar."""
    today = datetime.now()
    periods = {
        "Day": today - timedelta(days=1),
        "Week": today - timedelta(weeks=1),
        "Month": today - timedelta(days=30),
        "Year to Date": datetime(today.year, 1, 1),
        "Year (Trailing 12 Months)": today - timedelta(days=365),
        "2 Years": today - timedelta(days=730),
        "3 Years": today - timedelta(days=1095),
        "All Available Data": None,
        "Custom": "custom"
    }

    selected_period = st.sidebar.selectbox("Select Time Period", list(periods.keys()))

    if selected_period == "Custom":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start Date", today - timedelta(days=30))
        with col2:
            end_date = st.date_input("End Date", today)
        start_date = datetime.combine(start_date, datetime.min.time())
        end_date = datetime.combine(end_date, datetime.max.time())
    elif selected_period == "All Available Data":
        return None, None
    else:
        end_date = datetime.combine(today.date(), datetime.max.time())
        start_date = datetime.combine(periods[selected_period].date(), datetime.min.time())

    return start_date, end_date

def check_volume_condition(df, current_idx):
    """Check volume conditions for trade entry."""
    if current_idx < 6:
        return False
    current_volume = df.iloc[current_idx]['volume']
    previous_6_candles = df.iloc[current_idx - 6:current_idx]
    red_candles_volume = previous_6_candles[previous_6_candles['is_red']]['volume']
    return current_volume > red_candles_volume.max() if len(red_candles_volume) > 0 else True

def check_prior_6_opens(df, current_idx):
    """Check if current close is higher than previous 6 opens."""
    if current_idx < 6:
        return False
    current_close = df.iloc[current_idx]['close']
    previous_6_opens = df.iloc[current_idx - 6:current_idx]['open']
    return all(current_close > prev_open for prev_open in previous_6_opens)

def backtest_strategy(df):
    """Run the trading strategy backtest."""
    positions = []
    current_shares = 0
    consecutive_red = 0

    for i in range(len(df)):
        if i < 6:
            positions.append(0)
            continue

        current_candle = df.iloc[i]

        if current_candle['is_red']:
            consecutive_red += 1
        else:
            consecutive_red = 0

        if current_shares > 0:
            if consecutive_red >= 3 or (consecutive_red >= 2 and current_candle['close'] < current_candle['9ema']):
                current_shares = 0
            elif not current_candle['is_red'] and current_shares < 300 and current_candle['close'] > current_candle['9ema']:
                current_shares += 100

        elif current_shares == 0:
            if (not current_candle['is_red'] and
                check_prior_6_opens(df, i) and
                current_candle['close'] > current_candle['9ema'] and
                check_volume_condition(df, i)):
                current_shares = 100

        positions.append(current_shares)

    df['position'] = positions
    return df

def calculate_trade_metrics(results):
    """Calculate detailed trade metrics."""
    trade_changes = results[results['position'] != results['position'].shift(1)].copy()
    trade_changes['trade_type'] = np.where(trade_changes['position'] > trade_changes['position'].shift(1), 'entry', 'exit')

    trades = []
    current_entry = None

    for idx, row in trade_changes.iterrows():
        if row['trade_type'] == 'entry':
            current_entry = row
        elif row['trade_type'] == 'exit' and current_entry is not None:
            pnl = (row['execution_price'] - current_entry['execution_price']) * current_entry['position']  # Calculate P&L
            hold_time = (row['timestamp'] - current_entry['timestamp']).total_seconds() / 60  # in minutes
            trades.append({
                'entry_time': current_entry['timestamp'],
                'exit_time': row['timestamp'],
                'hold_time': hold_time,
                'pnl': pnl,
                'shares': current_entry['position'],
                'entry_price': current_entry['execution_price'],
                'exit_price': row['execution_price'],
            })

    if not trades:
        return pd.DataFrame()

    trades_df = pd.DataFrame(trades)
    return trades_df

def calculate_ratios(returns_series, risk_free_rate=0.02):
    """Calculate Sortino and Calmar ratios."""
    excess_returns = returns_series - (risk_free_rate / 252)  # Daily risk-free rate

    # Sortino Ratio
    negative_returns = returns_series[returns_series < 0]
    downside_std = np.sqrt(np.mean(negative_returns**2))
    sortino_ratio = (np.mean(excess_returns) * 252) / (downside_std * np.sqrt(252)) if downside_std != 0 else 0

    # Calmar Ratio
    max_drawdown = calculate_max_drawdown(returns_series)
    calmar_ratio = (np.mean(returns_series) * 252) / abs(max_drawdown) if max_drawdown != 0 else 0

    return sortino_ratio, calmar_ratio

def calculate_max_drawdown(returns_series):
    """Calculate maximum drawdown."""
    cum_returns = (1 + returns_series).cumprod()
    rolling_max = cum_returns.expanding(min_periods=1).max()
    drawdowns = cum_returns / rolling_max - 1
    return drawdowns.min()  # Returns as a fraction

def calculate_average_gain_loss(trades_df):
    """Calculate average gain and average loss in dollar and percentage terms."""
    gains = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] < 0]

    average_gain_size = gains['pnl'].mean() if not gains.empty else 0
    average_loss_size = losses['pnl'].mean() if not losses.empty else 0

    average_gain_pct = (average_gain_size / gains['entry_price'].mean() * 100) if not gains.empty else 0
    average_loss_pct = (average_loss_size / losses['entry_price'].mean() * 100) if not losses.empty else 0

    return average_gain_size, average_gain_pct, average_loss_size, average_loss_pct

def create_price_chart(data, trades_df, chart_type):
    """Create the price chart based on the chosen chart type."""
    fig = go.Figure()

    if data.empty:
        st.error("No data available to plot.")
        return fig

    if chart_type == "Line Chart":
        fig.add_trace(go.Scatter(
            x=data['timestamp'],
            y=data['close'],
            name='Price',
            line=dict(color='black')
        ))

    elif chart_type == "Bar Chart":
        # Use OHLC chart for the Bar Chart
        fig.add_trace(go.Ohlc(
            x=data['timestamp'],
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='OHLC'
        ))

    elif chart_type == "Candlestick Chart":
        fig = go.Figure(data=[go.Candlestick(
            x=data['timestamp'],
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Candlestick'
        )])

    # Integrate entry and exit points, check if trades_df is empty
    if not trades_df.empty:
        entries = trades_df[['entry_time', 'shares']].copy()
        entries['price'] = entries['entry_time'].map(data.set_index('timestamp')['close'])
        fig.add_trace(go.Scatter(
            x=entries['entry_time'],
            y=entries['price'],
            mode='markers',
            name='Entry',
            marker=dict(symbol='triangle-up', size=12, color='blue'),
            hovertemplate='Entry<br>Time: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))

        # Add exit points (blue triangles)
        exits = trades_df[['exit_time', 'shares']].copy()
        exits['price'] = exits['exit_time'].map(data.set_index('timestamp')['close'])
        fig.add_trace(go.Scatter(
            x=exits['exit_time'],
            y=exits['price'],
            mode='markers',
            name='Exit',
            marker=dict(symbol='triangle-down', size=12, color='blue'),  
            hovertemplate='Exit<br>Time: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))

    # Update layout
    fig.update_layout(
        title='Price Chart with Entry/Exit Points',
        title_font=dict(size=30),
        height=600,
        margin=dict(r=100),  # Add right margin for scroll area
        xaxis=dict(title='Date', rangeslider=dict(visible=False)),
        yaxis=dict(title='Price', side='left'),
        hovermode='x unified'
    )

    return fig

def format_currency(value):
    """Format the currency to thousands with two decimal places."""
    return f"${value / 1000:.2f}k"  # Convert to thousands

def main():
    # Sidebar for selecting stock dataset
    selected_stock = st.sidebar.radio("Select Stock Data", list(data_file_paths.keys()))

    start_date, end_date = get_date_range()

    # Load data based on selected stock
    data = load_and_prepare_data(data_file_paths[selected_stock])

    if not data.empty:
        # Filter data based on selected date range
        if start_date and end_date:
            data = data[(data['timestamp'] >= start_date) & 
                         (data['timestamp'] <= end_date)]

        # Run the backtest
        results = backtest_strategy(data)
        results['returns'] = (results['execution_price'].pct_change() * 
                              results['position'].shift(1) / 100)  # Division by 100 to account for shares

        # Calculate cumulative dollar returns using trade history
        trades_df = calculate_trade_metrics(results)

        # Initialize cumulative balance with zero, as starting balance isn't included
        cumulative_dollar_returns = []

        # Calculate total return from the trades DataFrame
        total_return = trades_df['pnl'].sum()  # Total dollar return

        cumulative_return = 0  # Initialize cumulative return for calculation
        for index, row in results.iterrows():
            if index >= 6:  # Skip the first 6 rows, as those cannot have trades yet
                if row['position'] != 0:
                    # Calculate P&L from trades when there is a position held
                    cumulative_return += row['returns'] * (row['position'] / 100)  # Apply the returns on the number of shares
                cumulative_dollar_returns.append(cumulative_return)  # Only keep the total return
            else:
                cumulative_dollar_returns.append(0)  # For the first few rows

        results['cumulative_dollar_returns'] = cumulative_dollar_returns

        # Check and calculate additional metrics if trades_df is not empty
        if not trades_df.empty:
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]

            # Calculate additional metrics
            avg_gain_size, avg_gain_pct, avg_loss_size, avg_loss_pct = calculate_average_gain_loss(trades_df)
            largest_win = winning_trades['pnl'].max() if not winning_trades.empty else 0
            largest_loss = losing_trades['pnl'].min() if not losing_trades.empty else 0

            # Calculate hold times in minutes
            avg_hold_time_winning = winning_trades['hold_time'].mean() if not winning_trades.empty else 0
            avg_hold_time_losing = losing_trades['hold_time'].mean() if not losing_trades.empty else 0

            sortino_ratio, calmar_ratio = calculate_ratios(results['returns'])

            # Prepare the metrics dictionary for structured display
            metrics_dict = {
                "Overall Performance": [
                    f"Total $ Return: {format_currency(total_return)}",
                    f"Max Drawdown: {format_currency(abs(calculate_max_drawdown(results['returns'])) * total_return)}",
                    f"Win Rate: {(len(winning_trades) / len(trades_df)) * 100:.2f}%",
                    f"Average Trade P&L: {format_currency(trades_df['pnl'].mean())}"
                ],
                "Win Metrics": [
                    "Average Gain (%)", 
                    "Average Gain ($)", 
                    "Largest Win ($)", 
                    "Average Hold Time (m)"
                ],
                "Value": [
                    f"{avg_gain_pct:.2f}%", 
                    format_currency(avg_gain_size), 
                    format_currency(largest_win), 
                    f"{avg_hold_time_winning:.2f}"
                ],
                "Loss Metrics": [
                    "Average Loss (%)", 
                    "Average Loss ($)", 
                    "Largest Loss ($)", 
                    "Average Hold Time (m)"
                ],
                "Loss Value": [
                    f"{avg_loss_pct:.2f}%", 
                    format_currency(avg_loss_size), 
                    format_currency(largest_loss), 
                    f"{avg_hold_time_losing:.2f}"
                ],
                "Risk Metrics": [
                    "Sortino Ratio", 
                    "Calmar Ratio"
                ],
                "Risk Value": [
                    f"{sortino_ratio:.2f}", 
                    f"{calmar_ratio:.2f}"
                ]
            }

            # Create a DataFrame from the structured metrics dictionary without row numbering
            metrics_rows = []
            max_length = max(len(metrics_dict["Overall Performance"]), 
                             len(metrics_dict["Win Metrics"]), 
                             len(metrics_dict["Loss Metrics"]), 
                             len(metrics_dict["Risk Metrics"]))

            for i in range(max_length):
                row = {
                    "Overall Performance": metrics_dict["Overall Performance"][i] if i < len(metrics_dict["Overall Performance"]) else "",
                    "Win Metrics": metrics_dict["Win Metrics"][i] if i < len(metrics_dict["Win Metrics"]) else "",
                    "Win Value": metrics_dict["Value"][i] if i < len(metrics_dict["Value"]) else "",
                    "Loss Metrics": metrics_dict["Loss Metrics"][i] if i < len(metrics_dict["Loss Metrics"]) else "",
                    "Loss Value": metrics_dict["Loss Value"][i] if i < len(metrics_dict["Loss Value"]) else "",
                    "Risk Metrics": metrics_dict["Risk Metrics"][i] if i < len(metrics_dict["Risk Metrics"]) else "",
                    "Risk Value": metrics_dict["Risk Value"][i] if i < len(metrics_dict["Risk Value"]) else "",
                }
                metrics_rows.append(row)

            # Create a DataFrame to display organized metrics
            metrics_df = pd.DataFrame(metrics_rows)

            # Display the metrics DataFrame as a table with bold headers
            st.markdown("<h2 style='font-size: 24px;'>Quantitative Performance Metrics</h2>", unsafe_allow_html=True)
            st.table(metrics_df.style.set_table_attributes('style="font-size: 20px; text-align: center;"').set_table_styles(
                [{'selector': 'th', 'props': [('font-weight', 'bold')]}]  # Adds bold to header
            ))

            # Performance visualizations
            st.subheader("Performance Analysis")
            col1, col2 = st.columns(2)

            with col1:
                # Trading performance chart (cumulative P&L starting from zero)
                if not trades_df.empty:
                    # Calculate cumulative P&L
                    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()

                    fig_pnl = go.Figure()
                    fig_pnl.add_trace(go.Scatter(
                        x=trades_df['entry_time'],
                        y=trades_df['cumulative_pnl'],  # Calculate cumulative P&L
                        mode='lines+markers',
                        name='Cumulative P&L',
                        line=dict(color='blue')
                    ))

                    # Set y-axis buffer to accommodate negative P&L
                    lower_bound_pnl = trades_df['cumulative_pnl'].min() - 1000
                    upper_bound_pnl = trades_df['cumulative_pnl'].max() * 1.2  # 20% above max cumulative P&L

                    # Update layout
                    fig_pnl.update_layout(
                        title='Cumulative Profit and Loss (P&L) Chart',
                        xaxis_title='Time',
                        yaxis_title='Cumulative P&L ($)',
                        height=600,
                        width=2000,
                        yaxis=dict(range=[lower_bound_pnl, upper_bound_pnl]),  # Ensure y-axis starts at an appropriate range
                        hovermode='x unified'
                    )
                    fig_pnl.update_layout(title_font=dict(size=25))
                    st.plotly_chart(fig_pnl, use_container_width=True)

            with col2:
                # Win/Loss distribution
                win_loss_data = pd.DataFrame({
                    'Category': ['Wins', 'Losses'],
                    'Count': [len(winning_trades), len(losing_trades)]
                })
                fig_pie = px.pie(win_loss_data, values='Count', names='Category',
                                title='Win/Loss Distribution',
                                color='Category', 
                                color_discrete_map={'Wins': 'green', 'Losses': 'red'})

                # Update pie chart font sizes and reduce padding
                fig_pie.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    textfont=dict(size=15)
                )
                fig_pie.update_layout(
                    title=dict(
                        text='Win/Loss Distribution',
                        font=dict(size=30)
                    ),
                    font=dict(size=25)
                )
                st.plotly_chart(fig_pie, use_container_width=400)

            # New Section for Performance Analysis by Day and Hour
            st.subheader("Performance Analysis by Day and Hour")
            col1, col2 = st.columns(2)

            # Toggle for return type
            return_type = st.radio("Select return type:", ("Dollar Returns", "Percentage Returns"), key="return_type")

            # Choose the strategy returns based on the type selected
            results['strategy_returns'] = np.where(return_type == "Dollar Returns", 
                results['returns'] * results['execution_price'].shift(1),  # Dollar returns calculation
                results['returns'] * 100)  # Convert to percentage

            with col1:
                # Day of week performance
                results['day_of_week'] = results['timestamp'].dt.day_name()
                day_performance = results.groupby('day_of_week')['strategy_returns'].sum().reset_index()

                # Set colors based on the sign of the returns
                day_performance['color'] = day_performance['strategy_returns'].apply(lambda x: 'green' if x >= 0 else 'red')

                fig_dow = px.bar(day_performance, 
                                 x='strategy_returns', 
                                 y='day_of_week',
                                 orientation='h', 
                                 title="Performance by Day of Week",
                                 color='color',  # Use the new 'color' column
                                 color_discrete_map={'green': 'green', 'red': 'red'})  # Map colors
                
                # Set title font size
                fig_dow.update_layout(title_font=dict(size=25))

                st.plotly_chart(fig_dow, use_container_width=True)

            with col2:
                # Hour performance
                results['hour'] = results['timestamp'].dt.hour
                hour_performance = results.groupby('hour')['strategy_returns'].sum().reset_index()

                # Set colors based on the sign of the returns
                hour_performance['color'] = hour_performance['strategy_returns'].apply(lambda x: 'green' if x >= 0 else 'red')

                fig_hour = px.bar(hour_performance, 
                                  x='strategy_returns', 
                                  y='hour',
                                  orientation='h', 
                                  title="Performance by Hour",
                                  color='color',  # Use the new 'color' column
                                  color_discrete_map={'green': 'green', 'red': 'red'})  # Map colors

                # Set title font size
                fig_hour.update_layout(title_font=dict(size=25))

                st.plotly_chart(fig_hour, use_container_width=True)

            # Create two columns for trade history and price chart
            col1, col2 = st.columns(2)

            with col1:
                # Trade history table
                st.subheader("Trade History")
                trades_df['entry_price'] = trades_df['entry_price'].apply(lambda x: f"${x:.2f}")
                trades_df['exit_price'] = trades_df['exit_price'].apply(lambda x: f"${x:.2f}")

                # Reorder columns for display
                trades_df = trades_df[['entry_time', 'entry_price', 'exit_time', 'exit_price', 'hold_time', 'pnl', 'shares']]

                # Apply custom styles to increase font size
                styled_df = trades_df.style.set_properties(**{'font-size': '20px'})

                # Display the styled DataFrame
                st.dataframe(styled_df, height=600)  # Set height to your preference

            with col2:
                # Chart Type Selection
                chart_type = st.selectbox("Select Chart Type", ["Line Chart", "Bar Chart", "Candlestick Chart"])

                # Create price chart with entry/exit signals
                fig_prices = create_price_chart(data, trades_df, chart_type)

                st.plotly_chart(fig_prices)

if __name__ == "__main__":
    main()