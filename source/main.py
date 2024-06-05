import streamlit as st
from pathlib import Path
import pandas as pd
import pandas_ta as ta
from lightweight_charts.widgets import StreamlitChart
import hashlib

from utils import resample_data, find_patterns, select_patterns, generate_target_price


# # Predefined special code
# SPECIAL_CODE = "pass"

# # Check for authentication
# if 'authenticated' not in st.session_state:
#     st.session_state.authenticated = False

# if not st.session_state.authenticated:
#     code = st.text_input("Enter the special code to access the app:", type="password")
#     if st.button("Submit"):
#         if code == SPECIAL_CODE:
#             st.session_state.authenticated = True
#             st.success("Authentication successful!")
#         else:
#             st.error("Incorrect code. Please try again.")
#     st.stop()  # Stop the app here if not authenticated

# Variables
source_timeframe = "4H"
oos_date_start = "2023-01-01"
hold_period = 1

# Load the data
cwd = Path.cwd()
data_path = cwd / 'data' / 'clean' / 'btcusdt.parquet'
raw_df = pd.read_parquet(data_path)
raw_df['time'] = pd.to_datetime(raw_df['time'])
features = pd.DataFrame({
    'time': raw_df['time'],
    'EMA_50': ta.ema(raw_df['close'], 50),
    'SMA_200': ta.sma(raw_df['close'], 200),
    # 'RSI' : ta.rsi(df['close'], 14),
}).dropna()

def compute_features(df):
    return pd.DataFrame({
    'time': df['time'],
    'EMA_50': ta.ema(df['close'], 50),
    'SMA_200': ta.sma(df['close'], 200),
    # 'RSI' : ta.rsi(df['close'], 14),
    }).dropna()

def string_to_color(s):
    # Generate a hash from the string
    hash_object = hashlib.md5(s.encode())
    hex_dig = hash_object.hexdigest()
    
    # Use the first 6 characters of the hash as the color
    color = f"#{hex_dig[:6]}"
    return color

# Perform pattern analysis
def run_analysis(data, source_timeframe, oos_date_start, hold_period, success_rate_threshold, total_count_threshold):
    # Resample the data
    dff = resample_data(data, source_timeframe, 'time')

    # Find patterns
    pattern_labels, count_totals, count_success = find_patterns(dff, hold_period, oos_date_start)

    # Find patterns that achieve over 70% success rate, and the total count is more than 50
    selected_patterns = select_patterns(count_totals, count_success, success_rate_threshold, total_count_threshold)
 
    # Generate the target price columns based on the selected patterns
    df = generate_target_price(dff, selected_patterns, pattern_labels, hold_period)

    # Return a list of dates for each occurrence
    setups = []
    for i in df[df['selected']].index:
        start_index = i + 1
        end_index = i + hold_period + 1

        if (start_index > len(df) - 1) or (end_index > len(df) - 1):
            break

        setups.append((df.loc[start_index, 'time'], df.loc[end_index, 'time'], df.loc[i, 'targets']))

    return df, setups

# Display the passed dataframe
def plot_chart(data, features, **kwargs):
    df = data.copy()

    # Preprocess dataframe
    df['time'] = pd.to_datetime(df['time'])

    # Instantiate the chart object
    main_chart = StreamlitChart(width=1200, height=800)  # Fixed size as fallback
    main_chart.legend(visible=True)
    main_chart.set(df)

    if kwargs.get('start_date', None):
        main_chart.watermark(f"{start_date}", 10)

    # Create line plots for selected features (excluding RSI)
    for feature in features.columns:
        if feature == 'target':
            line = main_chart.create_line(feature, color='#FF0000',  width=2,  price_line=True)  # Label must match the column of the values
            line.set(features[['time', feature]])

        elif feature != 'time' and feature != 'RSI':
            line = main_chart.create_line(feature, color=string_to_color(feature))  # Label must match the column of the values
            line.set(features[['time', feature]])

    # Display the charts
    main_chart.load()

    if 'RSI' in features:
        rsi_chart = StreamlitChart(width=1200, height=200)  # Separate pane for RSI
        rsi_chart.legend(visible=True)
        rsi_line = rsi_chart.create_line('RSI')
        rsi_line.set(features[['time', 'RSI']])
        rsi_chart.load()

st.set_page_config(layout="wide", initial_sidebar_state='collapsed')  # Set layout to wide mode

# Sidebar inputs
with st.sidebar:
    st.header("Settings")

    st.subheader("General Parameters")
    timeframe_options = ['5MIN', '15MIN', '30MIN', '1H', '2H', '4H', '8H', '1D', '1W']

    # test_timeframe = st.selectbox("Chart Timeframe", ["1MIN"] + timeframe_options, index=0)
    source_timeframe = st.selectbox("Pattern Source Timeframe", timeframe_options, index=7)
    oos_date_start = st.date_input("OOS Start Date", value=pd.to_datetime("2023-01-01").date())

    st.subheader("Pattern Selection Parameters")
    hold_period = st.number_input("Hold Period", value=1, min_value=1, max_value=50)
    success_rate_threshold = st.number_input("Success Rate Threshold % ", value=70, min_value=1, max_value=100, step=1) / 100
    total_count_threshold = st.number_input("Total Count Threshold", value=50, min_value=1, max_value=1000)
    

# Ensure start date is not after end date
if True:
    # Run the analysis
    df, setups = run_analysis(raw_df, source_timeframe, oos_date_start, hold_period, success_rate_threshold, total_count_threshold)

    if len(setups) == 0:
        st.write("No setups found.")
        
    else:
        # Initialize session state for setup index
        if 'setup_index' not in st.session_state:
            st.session_state.setup_index = 0

        # Place buttons side by side using st.columns
        col1, col2, col3 = st.columns([0.1, 0.6, 0.3])

        with col1:
            st.write("")  # Placeholder to create spacing
            test_timeframe = st.selectbox("Chart Timeframe", ["1MIN"] + timeframe_options, index=1)

        with col2:
            st.write("")

        with col3:
            button_prev, button_next = st.columns([0.5, 0.5])
            
            with button_prev:
                if st.button("Previous Setup"):
                    st.session_state.setup_index = (st.session_state.setup_index - 1) % len(setups)

            with button_next:
                if st.button("Next Setup"):
                    st.session_state.setup_index = (st.session_state.setup_index + 1) % len(setups)

         # Generate test dataframe
        test_df = resample_data(raw_df, test_timeframe, 'time')
        features = compute_features(test_df)

        # Get the current setup
        current_setup = setups[st.session_state.setup_index]
        start_date, end_date, target_price = current_setup

        # st.write(f"DATE : {start_date} to {end_date} | f"TARGET: {target_price}"")

        # Filter the dataframe based on the selected dates
        filtered_df = test_df[(test_df['time'] >= start_date) & (test_df['time'] < end_date)]
        filtered_features = features[(features['time'] >= start_date) & (features['time'] < end_date)]
        filtered_features.loc[:, 'target'] = target_price  # Add target price to features dataframe

        # Use Streamlit container to adjust chart to screen size
        with st.container():
            plot_chart(filtered_df, filtered_features, start_date=start_date, end_date=end_date, target=target_price)
