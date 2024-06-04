import pandas as pd

from pathlib import Path
from typing import List, Dict


# CLASSES
class Signal:
    """
    A class to represent a trading signal.

    Attributes:
    - pattern (str): The pattern code.
    - target (int): The target count.
    - success_rate (float): The success rate of the pattern.
    """

    def __init__(self, pattern: str, target: int, success_rate: float):
        """
        Constructs all the necessary attributes for the Signal object.

        Parameters:
        - pattern (str): The pattern code.
        - target (int): The target count.
        - success_rate (float): The success rate of the pattern.
        """
        self.pattern = pattern
        self.target = target
        self.success_rate = success_rate
    
    def __repr__(self):
        return f"Signal(pattern={self.pattern}, target={self.target}, success_rate={self.success_rate})"
    
    def __gt__(self, other):
        return self.success_rate > other.success_rate
    
    def __lt__(self, other):
        return self.success_rate < other.success_rate
    
    def __eq__(self, other):
        return self.success_rate == other.success_rate


# FUNCTIONS
def read_asset_data(asset_name: str) -> pd.DataFrame:
    """
    Reads asset data from a parquet file.

    Parameters:
    - asset_name (str): The name of the asset.

    Returns:
    - pd.DataFrame: DataFrame containing the asset data if the file exists, otherwise an empty DataFrame.
    """
    assert isinstance(asset_name, str), "asset_name must be a string."

    # Convert asset name to lowercase
    asset_name = asset_name.lower()
    
    # Construct the path to the parquet file
    parquet_path = Path(f"data/clean/{asset_name}.parquet")
    
    # Check if the file exists and read the parquet file
    if Path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    else:
        return pd.DataFrame()
    

def resample_data(data: pd.DataFrame, timeframe: str, datetime_col: str = None) -> pd.DataFrame:
    """
    Resamples 1-minute OHLC data into a given timeframe.
    
    Parameters:
    - data (pd.DataFrame): A DataFrame containing 1-minute OHLC data with a DatetimeIndex or a datetime column.
                           The columns should be ['open', 'high', 'low', 'close'].
    - timeframe (str): The resampling timeframe, e.g., '5T', '15T', '30T', '1H', '2H', '4H', '8H', '1D', '1W'.
    - datetime_col (str): The name of the column to use as the datetime index if the index is not a DatetimeIndex.
    
    Returns:
    - resampled_data (pd.DataFrame): A DataFrame with the resampled OHLC data.
    """
    # Create a copy of the data to avoid modifying the original DataFrame
    data = data.copy()

    # Ensure the timeframe is in uppercase
    timeframe = timeframe.lower()
    
    # Check if the DataFrame has the correct columns
    required_columns = ['open', 'high', 'low', 'close']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Input data must contain the following columns: {required_columns}")
    
    # If datetime_col is provided, set it as the index
    if datetime_col:
        if datetime_col not in data.columns:
            raise ValueError(f"The specified datetime column '{datetime_col}' is not in the DataFrame.")
        data[datetime_col] = pd.to_datetime(data[datetime_col])
        data.set_index(datetime_col, inplace=True)
    
    # Check if the index is a DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("Index of the DataFrame must be a DatetimeIndex or a valid datetime column must be provided.")
    
    # Define the aggregation dictionary for resampling
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }
    
    # Perform the resampling
    resampled_data = data.resample(timeframe).apply(ohlc_dict).dropna(how='any')
    resampled_data.reset_index(inplace=True, names=[datetime_col])
    
    return resampled_data


def compare_values(value_1: float, value_2: float) -> str:
    """
    Compares two float values and returns '1' if the first value is greater, else returns '0'.
    
    Parameters:
    - value_1 (float): The first value to compare.
    - value_2 (float): The second value to compare.
    
    Returns:
    - (str): '1' if value_1 is greater than value_2, otherwise '0'.
    """
    if value_1 > value_2:
        return '1'
    else:
        return '0'


def filled(price: float, high: float, low: float) -> bool:
    """
    Checks if a price is within the high and low range.

    Parameters:
    - price (float): The price to check.
    - high (float): The high range value.
    - low (float): The low range value.

    Returns:
    - (bool): True if price is within the range, False otherwise.
    """
    return (price <= high) and (price >= low)


def find_patterns(df: pd.DataFrame, hold_period: int, oos_date_start: str) -> pd.DataFrame:
    """
    Identifies and counts specific patterns in OHLC data over a given holding period.

    Parameters:
    - df (pd.DataFrame): DataFrame containing OHLC data with columns ['time', 'open', 'high', 'low', 'close'].
    - hold_period (int): The holding period to calculate rolling highest and lowest prices.
    - oos_date_start (str): The out-of-sample start date to filter the DataFrame.

    Returns:
    - pattern_labels (list): List of pattern labels for each row in the DataFrame.
    - count_totals (dict): Dictionary with counts of each pattern found.
    - count_success (dict): Dictionary with success counts for each pattern in different categories.
    """
    # Store patterns and their statistics
    count_totals = {}
    count_success = {}
    pattern_labels = ["0"] * len(df)

    df = df.copy()

    # Calculate rolling highest and lowest prices
    df['highest'] = df['high'].rolling(hold_period).max()
    df['lowest'] = df['low'].rolling(hold_period).min()

    # Filter DataFrame for in-sample data
    df = df[df['time'] < oos_date_start]

    # Iterate through the DataFrame
    for i in range(hold_period + 1, len(df)):
        bar_0 = df.iloc[i - hold_period]
        bar_1 = df.iloc[i - hold_period - 1]

        _highest = df['highest'].iloc[i]
        _lowest = df['lowest'].iloc[i]

        # Generate pattern string
        _pattern = compare_values(bar_0['high'], bar_1['high']) + \
                   compare_values(bar_0['low'], bar_1['low']) + \
                   compare_values(bar_0['open'], bar_1['open']) + \
                   compare_values(bar_0['close'], bar_1['close']) + \
                   compare_values(bar_0['open'], bar_1['close']) + \
                   compare_values(bar_0['close'], bar_1['open'])
        
        # Initialize pattern in the dictionaries if not present
        if _pattern not in count_totals:
            count_totals[_pattern] = 0
            count_success[_pattern] = {
                'open': 0,
                'high': 0,
                'low': 0,
                'open_1': 0,
                'high_1': 0,
                'low_1': 0,
                'close_1': 0
            }

        # Update the count of the pattern
        count_totals[_pattern] += 1
        
        # Check and update success counts
        if filled(bar_0['open'], _highest, _lowest):
            count_success[_pattern]['open'] += 1
        if filled(bar_0['high'], _highest, _lowest):
            count_success[_pattern]['high'] += 1
        if filled(bar_0['low'], _highest, _lowest):
            count_success[_pattern]['low'] += 1
        if filled(bar_1['open'], _highest, _lowest):
            count_success[_pattern]['open_1'] += 1
        if filled(bar_1['high'], _highest, _lowest):
            count_success[_pattern]['high_1'] += 1
        if filled(bar_1['low'], _highest, _lowest):
            count_success[_pattern]['low_1'] += 1
        if filled(bar_1['close'], _highest, _lowest):
            count_success[_pattern]['close_1'] += 1
        
        # Add the pattern to the list of labels
        pattern_labels[i] = _pattern

    return pattern_labels, count_totals, count_success


def select_patterns(count_totals: Dict[str, int], count_success: Dict[str, Dict[str, int]], 
                    success_rate_threshold: float = 0.80, total_count_threshold: int = 100) -> List[Signal]:
    """
    Selects pattern codes and targets that achieve over a specified success rate and total count.

    Parameters:
    - count_totals (Dict[str, int]): Dictionary with total counts of each pattern.
    - count_success (Dict[str, Dict[str, int]]): Dictionary with success counts for each pattern in different categories.
    - success_rate_threshold (float): The minimum success rate required to select a pattern. Default is 0.80.
    - total_count_threshold (int): The minimum total count required to select a pattern. Default is 100.

    Returns:
    - selected_patterns (List[Signal]): A list of Signal namedtuples containing selected patterns with their success rates.
    """
    if not (set(count_totals.keys()) == set(count_success.keys())):
        raise ValueError("Keys in count_totals and count_success do not match.")
    
    selected_patterns = []

    for pattern in count_totals.keys():
        if count_totals[pattern] > total_count_threshold:
            for target in count_success[pattern].keys():
                success_rate = count_success[pattern][target] / count_totals[pattern]
                if success_rate > success_rate_threshold:
                    signal = Signal(
                        pattern=pattern,
                        target=target,
                        success_rate=success_rate
                    )
                    selected_patterns.append(signal)
    
    # Sort the selected patterns by success rate in descending order
    selected_patterns.sort(key=lambda x: x.success_rate, reverse=True)
    
    return selected_patterns



if __name__ == "__main__":
    print(read_asset_data('btcusdt'))