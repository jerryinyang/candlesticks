from pathlib import Path
import numpy as np
import pandas as pd
import pandas_ta as ta
from lightweight_charts import Chart
from lightweight_charts.table import Table
import hashlib
from utils import resample_data, find_patterns, select_patterns, generate_target_price, SessionState


# Variables
oos_date_start = "2023-01-01"

# Load the data
cwd = Path.cwd()
data_path = cwd / 'data' / 'clean' / 'btcusdt.parquet'
raw_df = pd.read_parquet(data_path)
raw_df['time'] = pd.to_datetime(raw_df['time'])

chart : Chart = None
table_config : Table = None
table_setups : Table = None


session_state = SessionState(
    setups = None,
    current_setup_index=-1
)


def compute_features(df, target_price : float):

    return pd.DataFrame({
    'time': df['time'],
    'target': np.full(len(df), target_price),
    'EMA_50': ta.ema(df['close'], 50),
    'SMA_200': ta.sma(df['close'], 200),
    # 'RSI' : ta.rsi(df['close'], 14),
    })


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

    return setups, selected_patterns


def set_configuration(row, *args):
    cell_clicked : str = args[0]
    row_name = row['#']

    if cell_clicked == '# -':
        if row_name == 'Threshold Success Rate % ':
            row['# --'] = str(max(1, int(row['# --']) - 1))
        elif row_name == 'Threshold Total Count':
            row['# --'] = str(max(1, int(row['# --']) - 5))

    elif cell_clicked == '# +':
        if row_name == 'Threshold Success Rate % ':
            row['# --'] = str(min(100, int(row['# --']) + 1))
        elif row_name == 'Threshold Total Count':
            row['# --'] = str(min(1000, int(row['# --']) + 5))

    elif cell_clicked == '# --':
        if row_name == 'Threshold Success Rate % ':
            row['# --'] = "70"
        elif row_name == 'Threshold Total Count':
            row['# --'] = "50"


def fetch_setups(chart : Chart):
    
    # Get Inputs
    source_timeframe = str(chart.topbar.get('source_timeframe').value).upper()
    hold_period = int(chart.topbar.get('hold_period').value)

    # Get the configuration
    success_rate_threshold = int(table_config.get(list(table_config.keys())[0])['# --']) / 100
    total_count_threshold = int(table_config.get(list(table_config.keys())[1])['# --'])

    # Run the analysis
    setups, signals = run_analysis(raw_df, source_timeframe, oos_date_start, hold_period, success_rate_threshold, total_count_threshold)

    # Update the global setups
    session_state.setups = setups

    table_setups.clear()
    for setup in signals:
        table_setups.new_row(
            *(str(setup.pattern), str(setup.success_rate), str(setup.target))
        )

    # Display the setup
    display_setup(chart)


def display_setup(chart : Chart):
    if session_state.setups is None:
        fetch_setups(chart)

    setups = session_state.setups
    setup_index = min(session_state.current_setup_index + 1,  len(setups) - 1)

    if len(setups) == 0:
        print("No setups found.")
        chart.topbar["title_setup_count"].set("0 setups found")
        return

    # Get chart timeframe value
    chart_timeframe = str(chart.topbar.get('timeframe').value).upper()

    # Get the current setup
    current_setup = setups[setup_index]
    start_date, end_date, target_price = current_setup

    # Generate test dataframe
    test_df = resample_data(raw_df, chart_timeframe, 'time')
    features = compute_features(test_df, target_price)

    # Filter the dataframe based on the selected dates
    filtered_df = test_df[(test_df['time'] >= start_date) & (test_df['time'] < end_date)]
    # filtered_features = features[features['time'].isin(filtered_df['time'])]
    filtered_features = features[(features['time'] >= start_date) & (features['time'] < end_date)]

    # Use Streamlit container to adjust chart to screen size
    plot_chart(chart, filtered_df, filtered_features, start_date=start_date, end_date=end_date, target=target_price)

    chart.topbar["title_setup_count"].set(f"{(setup_index % len(setups)) + 1} of {len(setups)}")


# Display the passed dataframe
def plot_chart(chart : Chart, data, features, **kwargs):
    df = data.copy()

    # Preprocess dataframe
    df['time'] = pd.to_datetime(df['time'])

    lines = {line.name : line for line in  chart.lines()}

    # Create line plots for the target price and selected features (excluding RSI)
    for feature in features.columns:
        line = lines.get(feature, None)
        if line is None:
            # Create a fixed horizontal line for the target price
            if feature == 'target':
                line = chart.create_line(feature, color='#FF0000',  width=2,  price_line=True)  # Label must match the column of the values
            # Plot other features
            elif feature != 'time' and feature != 'RSI':
                line = chart.create_line(feature, color=string_to_color(feature))  # Label must match the column of the values
            else:
                continue

        line.set(features[['time', feature]])
        

    # Display the charts
    chart.set(df)
    chart.fit()

    # if 'RSI' in features:
    #     rsi_chart = Chart()  # New Pane
    #     rsi_chart.legend(visible=True)
    #     rsi_line = rsi_chart.create_line('RSI')
    #     rsi_line.set(features[['time', 'RSI']])
    #     rsi_chart.load()


def on_timeframe_selection(chart : Chart):
    display_setup(chart)


def on_reset(chart):
    session_state.setups = None
    session_state.current_setup_index = -1

    fetch_setups(chart)


def on_next_setup(chart):
    session_state.current_setup_index += 1
    display_setup(chart)


def on_previous_setup(chart):
    session_state.current_setup_index -= 1
    display_setup(chart)


def on_nothing(*args, **kwargs):
    print("args", args)
    print("kwargs", kwargs)


options_source_timeframe = ['1min', '3min', '5min', '15min', '30min', '1h', '2h', '4h', '8h', '1d', '1w']
configs = [
    ("Threshold Success Rate % ", '-', '75', '+'),
    ("Threshold Total Count", '-', '50', '+'),
]

if __name__ == '__main__':
    chart = Chart(inner_width=.7, inner_height=1, maximize=True, toolbox=True)

    # Chart Top Bar Configurations
    chart.topbar.menu(
        'timeframe', 
        options=(options_source_timeframe),
        align='left',
        func=on_timeframe_selection
    )

    chart.topbar.button(
        'previous_setup',
        'Previous',
        align='left',
        func=on_previous_setup
    )

    chart.topbar.button(
        'next_setup',
        'Next',
        align='left',
        func=on_next_setup
    )

    chart.topbar.textbox(
        'title_setup_count',
        '# of #',
        align='left',
    )
    
    
    chart.topbar.textbox(
        'title_source_timeframe',
        'Source Timeframe',
        align='right'
    )
    chart.topbar.menu(
        'source_timeframe', 
        options=('4h', '8h', '1d', '1w'),
        align='right',
        func=fetch_setups,
    )
    
    chart.topbar.textbox(
        'title_hold_period',
        'Hold Period',
        align='right',
    )
    chart.topbar.textbox(
        'hold_period',
        '1',
        align='right',
        func=fetch_setups
    )
    
    chart.topbar.button(
        'reset_backtest',
        'Reset Backtest',
        align='right',
        func=on_reset
    )


    # Configuration Dashboard
    table_config = chart.create_table(
        width=0.3, 
        height=.5,
        headings=('#', '# -', '# --', '# +'),
        widths=(0.4, 0.1, 0.2, 0.1),
        alignments=('center', 'left', 'center', 'right'),
        position='right', 
        func=set_configuration, 
        return_clicked_cells=True
    )
    
    for config in configs:
        table_config.new_row(*config)

    table_setups = chart.create_table(
        width=0.3, 
        height=.5,
        headings=("Patterns", "Success Percentage", "Target"),
        widths=(0.4, 0.2, 0.2, 0.2),
        alignments=('center', 'center', 'center', 'center'),
        position='right', 
        func=on_nothing, 
    )

    fetch_setups(chart)

    chart.show(block=True)
    
