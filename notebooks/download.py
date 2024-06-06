import datetime
import gc
import os
from multiprocessing import Pool

import boto3
import ccxt
import pandas as pd

# AWS S3 setup
s3_client = boto3.client('s3')
s3_bucket_name = 'binancebucket1m'  # Replace with your S3 bucket name

# Connect to the Binance client
binance_client = ccxt.binance()
bybit_client = ccxt.bybit()

klines_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignored']

# Convert timeframe to amount of seconds
timeframes_to_seconds = {
    "1m": 60,
    "3m": 3 * 60,
    "5m": 5 * 60,
    "15m": 15 * 60,
    "30m": 30 * 60,
    "1h": 60 * 60,
    "2h": 2 * 60 * 60,
    "4h": 4 * 60 * 60,
    "6h": 6 * 60 * 60,
    "8h": 8 * 60 * 60,
    "12h": 12 * 60 * 60,
    "1d": 24 * 60 * 60,
    "3d": 3 * 24 * 60 * 60,
    "1w": 7 * 24 * 60 * 60,
    "1M": 30 * 24 * 60 * 60
}

def upload_file_to_s3(file_path, bucket, s3_key):
    """Upload a file to an S3 bucket.

    Args:
        file_path (str): Path to the file to upload
        bucket (str): Bucket to upload to
        s3_key (str): S3 object key

    Returns:
        bool: True if file was uploaded, else False
    """
    try:
        s3_client.upload_file(file_path, bucket, s3_key)
        return True
    except Exception as e:
        return False

def fetch_binance_data(symbol, timeframe, _from: datetime.datetime, to: datetime.datetime, batch_size=1000):
    print(f"Downloading {symbol}")
    try:
        df_list = []  # Store data for each symbol
        earliest = binance_client.parse8601('2017-08-17T00:00:00Z')

        if _from.timestamp() * 1000 < earliest:
            _from = datetime.datetime.fromtimestamp(earliest / 1000)

        amount_of_candles = (to - _from).total_seconds() / timeframes_to_seconds[timeframe]

        # Read from csv if it exists
        try:
            df = pd.read_csv(f'.cache/ohlcv/{symbol}/{symbol}_{timeframe}.csv')
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df[df['timestamp'] >= _from]
            df = df[df['timestamp'] <= to]
            if df.shape[0] >= amount_of_candles:
                return df
        except Exception:
            pass

        # Fetch data in batches
        while True:
            from_timestamp = int(_from.timestamp() * 1000)
            ohlcv = binance_client.public_get_klines({
                'symbol': symbol,
                'interval': timeframe,
                'limit': batch_size,
                'startTime': from_timestamp
            })
            if not ohlcv:  # End of data
                break

            if from_timestamp > to.timestamp() * 1000:
                break

            df = pd.DataFrame(ohlcv, columns=klines_columns)

            # Convert timestamp from string (ms). First remove the last 3 characters (ms) and then convert to unsigned long (int64)
            df['timestamp'] = df['timestamp'].str[:-3].astype('int64')
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

            df_list.append(df)

            _from = df['timestamp'].iloc[-1] + pd.Timedelta(1, unit='ms')

            # Save each batch to file
            symbol_folder = f'.cache/ohlcv/{symbol}'
            os.makedirs(symbol_folder, exist_ok=True)
            batch_file_path = f'{symbol_folder}/{symbol}_{timeframe}_{df["timestamp"].iloc[0].strftime("%Y%m%d%H%M%S")}.csv'
            df.to_csv(batch_file_path, index=False)

            # Upload batch file to S3
            upload_file_to_s3(batch_file_path, s3_bucket_name, f'ohlcv/{symbol}/{os.path.basename(batch_file_path)}')

            # Remove batch file from local storage to save memory
            os.remove(batch_file_path)

            # Clear DataFrame and trigger garbage collection
            df_list.clear()
            gc.collect()

        if len(df_list) == 0:
            return pd.DataFrame(columns=klines_columns)

        df_list = [df.dropna(how='all', axis=1) for df in df_list]

        df = pd.concat(df_list, ignore_index=True)
        df.drop_duplicates(subset='timestamp', inplace=True)
        df.sort_values('timestamp', inplace=True)

        print(f"Data for {symbol} has been completely added and uploaded to S3.")

        return df
    except Exception as e:
        return pd.DataFrame(columns=klines_columns)

def fetch_many_binance_data(symbols, timeframe, _from, to, batch_size):
    try:
        with Pool(processes=len(symbols)) as pool:
            args = [(symbol, timeframe, _from, to, batch_size) for symbol in symbols]
            result = pool.starmap(fetch_binance_data, args)
        total_df = pd.concat(result, ignore_index=True)
        return total_df
    except Exception as e:
        return pd.DataFrame()

def get_bybit_symbols():
    try:
        markets = bybit_client.load_markets()
        return [markets[market]['symbol'] for market in markets]
    except Exception:
        return []

def main(symbols, timeframe, from_date, to_date, batch_size=1000):
    """
    Main function to fetch Binance data for a list of symbols and upload it to AWS S3.

    Args:
        symbols (list): List of cryptocurrency symbols (e.g., ['BTCUSDT', 'ETHUSDT'])
        timeframe (str): Timeframe for the candlesticks (e.g., '1m', '1h', '1d')
        from_date (str): Start date for the data in 'YYYY-MM-DD' format
        to_date (str): End date for the data in 'YYYY-MM-DD' format
        batch_size (int): Number of OHLCV data points to fetch per API call
    """
    try:
        _from = datetime.datetime.strptime(from_date, '%Y-%m-%d')
        to = datetime.datetime.strptime(to_date, '%Y-%m-%d')

        total_df = fetch_many_binance_data(symbols, timeframe, _from, to, batch_size)

        if not total_df.empty:
            print("Data fetching and uploading to S3 completed successfully.")
    except Exception as e:
        pass

if __name__ == "__main__":
    try:
        # Initialize the Binance exchange
        binance = ccxt.binance()
        markets = binance.load_markets()
        all_pairs = [market.replace("/", '') for market in markets if market.endswith('/USDT')]

        old_pairs_str = "BINANCE:1000BONKUSDT.P,BINANCE:1000FLOKIUSDT.P,BINANCE:1000LUNCUSDT.P,BINANCE:1000PEPEUSDT.P,BINANCE:1000RATSUSDT.P,BINANCE:1000SATSUSDT,BINANCE:1000SATSUSDT.P,BINANCE:1000SHIBUSDT.P,BINANCE:1000XECUSDT.P,BINANCE:1INCHUSDT,BINANCE:1INCHUSDT.P,BINANCE:ACAUSDT,BINANCE:ACEUSDT.P,BINANCE:ACHUSDT,BINANCE:ACHUSDT.P,BINANCE:ADAUSDT,BINANCE:ADAUSDT.P,BINANCE:ADXUSDT,BINANCE:AERGOUSDT,BINANCE:AEVOUSDT,BINANCE:AEVOUSDT.P,BINANCE:AGIXUSDT,BINANCE:AGIXUSDT.P,BINANCE:AGLDUSDT,BINANCE:AGLDUSDT.P,BINANCE:AIUSDT,BINANCE:AIUSDT.P,BINANCE:AKROUSDT,BINANCE:ALGOUSDT,BINANCE:ALGOUSDT.P,BINANCE:ALICEUSDT.P,BINANCE:ALPACAUSDT,BINANCE:ALPHAUSDT,BINANCE:ALPHAUSDT.P,BINANCE:ALTUSDT,BINANCE:ALTUSDT.P,BINANCE:AMBUSDT,BINANCE:AMBUSDT.P,BINANCE:AMPUSDT,BINANCE:ANKRUSDT,BINANCE:ANKRUSDT.P,BINANCE:APEUSDT,BINANCE:APEUSDT.P,BINANCE:API3USDT.P,BINANCE:APTUSDT,BINANCE:APTUSDT.P,BINANCE:ARBUSDT,BINANCE:ARBUSDT.P,BINANCE:ARDRUSDT,BINANCE:ARKMUSDT,BINANCE:ARKMUSDT.P,BINANCE:ARKUSDT.P,BINANCE:ARPAUSDT,BINANCE:ARPAUSDT.P,BINANCE:ASTRUSDT,BINANCE:ASTRUSDT.P,BINANCE:ASTUSDT,BINANCE:ATAUSDT,BINANCE:ATAUSDT.P,BINANCE:ATOMUSDT.P,BINANCE:AUDIOUSDT,BINANCE:AVAXUSDT.P,BINANCE:AXLUSDT,BINANCE:AXLUSDT.P,BINANCE:AXSUSDT.P,BINANCE:BAKEUSDT,BINANCE:BAKEUSDT.P,BINANCE:BANDUSDT.P,BINANCE:BATUSDT,BINANCE:BATUSDT.P,BINANCE:BBUSDT,BINANCE:BBUSDT.P,BINANCE:BEAMXUSDT,BINANCE:BEAMXUSDT.P,BINANCE:BELUSDT,BINANCE:BELUSDT.P,BINANCE:BETAUSDT,BINANCE:BICOUSDT,BINANCE:BICOUSDT.P,BINANCE:BIGTIMEUSDT.P,BINANCE:BLURUSDT,BINANCE:BLURUSDT.P,BINANCE:BLZUSDT,BINANCE:BLZUSDT.P,BINANCE:BNTUSDT.P,BINANCE:BNXUSDT,BINANCE:BNXUSDT.P,BINANCE:BOMEUSDT,BINANCE:BOMEUSDT.P,BINANCE:BONKUSDT,BINANCE:BSWUSDT,BINANCE:BTTCUSDT,BINANCE:C98USDT,BINANCE:C98USDT.P,BINANCE:CAKEUSDT,BINANCE:CAKEUSDT.P,BINANCE:CELOUSDT,BINANCE:CELOUSDT.P,BINANCE:CELRUSDT,BINANCE:CELRUSDT.P,BINANCE:CFXUSDT,BINANCE:CFXUSDT.P,BINANCE:CHESSUSDT,BINANCE:CHRUSDT,BINANCE:CHRUSDT.P,BINANCE:CHZUSDT,BINANCE:CHZUSDT.P,BINANCE:CKBUSDT,BINANCE:CKBUSDT.P,BINANCE:CLVUSDT,BINANCE:COMBOUSDT.P,BINANCE:COSUSDT,BINANCE:COTIUSDT,BINANCE:COTIUSDT.P,BINANCE:CRVUSDT,BINANCE:CRVUSDT.P,BINANCE:CTSIUSDT,BINANCE:CTSIUSDT.P,BINANCE:CTXCUSDT,BINANCE:CVCUSDT,BINANCE:CYBERUSDT.P,BINANCE:DARUSDT,BINANCE:DARUSDT.P,BINANCE:DATAUSDT,BINANCE:DENTUSDT,BINANCE:DENTUSDT.P,BINANCE:DFUSDT,BINANCE:DGBUSDT,BINANCE:DOCKUSDT,BINANCE:DODOUSDT,BINANCE:DODOXUSDT.P,BINANCE:DOGEUSDT,BINANCE:DOGEUSDT.P,BINANCE:DOTUSDT.P,BINANCE:DUSKUSDT,BINANCE:DUSKUSDT.P,BINANCE:DYDXUSDT,BINANCE:DYDXUSDT.P,BINANCE:DYMUSDT.P,BINANCE:EDUUSDT,BINANCE:EDUUSDT.P,BINANCE:ENAUSDT,BINANCE:ENAUSDT.P,BINANCE:ENJUSDT,BINANCE:ENJUSDT.P,BINANCE:ENSUSDT.P,BINANCE:EOSUSDT,BINANCE:EOSUSDT.P,BINANCE:EPXUSDT,BINANCE:ETCUSDT.P,BINANCE:ETHFIUSDT,BINANCE:ETHFIUSDT.P,BINANCE:EURUSDT,BINANCE:FDUSDTRY,BINANCE:FDUSDUSDT,BINANCE:FETUSDT,BINANCE:FETUSDT.P,BINANCE:FIDAUSDT,BINANCE:FILUSDT,BINANCE:FILUSDT.P,BINANCE:FIOUSDT,BINANCE:FLMUSDT,BINANCE:FLMUSDT.P,BINANCE:FLOKIUSDT,BINANCE:FLOWUSDT,BINANCE:FLOWUSDT.P,BINANCE:FORUSDT,BINANCE:FRONTUSDT,BINANCE:FRONTUSDT.P,BINANCE:FTMUSDT,BINANCE:FTMUSDT.P,BINANCE:FUNUSDT,BINANCE:FXSUSDT,BINANCE:FXSUSDT.P,BINANCE:GALAUSDT,BINANCE:GALAUSDT.P,BINANCE:GALUSDT.P,BINANCE:GASUSDT.P,BINANCE:GFTUSDT,BINANCE:GLMRUSDT,BINANCE:GLMUSDT,BINANCE:GLMUSDT.P,BINANCE:GMTUSDT,BINANCE:GMTUSDT.P,BINANCE:GRTUSDT,BINANCE:GRTUSDT.P,BINANCE:GTCUSDT.P,BINANCE:HARDUSDT,BINANCE:HBARUSDT,BINANCE:HBARUSDT.P,BINANCE:HFTUSDT,BINANCE:HFTUSDT.P,BINANCE:HIFIUSDT,BINANCE:HIFIUSDT.P,BINANCE:HIGHUSDT.P,BINANCE:HOOKUSDT,BINANCE:HOOKUSDT.P,BINANCE:HOTUSDT,BINANCE:HOTUSDT.P,BINANCE:ICPUSDT.P,BINANCE:ICXUSDT,BINANCE:ICXUSDT.P,BINANCE:IDEXUSDT,BINANCE:IDUSDT,BINANCE:IDUSDT.P,BINANCE:IMXUSDT,BINANCE:IMXUSDT.P,BINANCE:INJUSDT.P,BINANCE:IOSTUSDT,BINANCE:IOSTUSDT.P,BINANCE:IOTAUSDT,BINANCE:IOTAUSDT.P,BINANCE:IOTXUSDT,BINANCE:IOTXUSDT.P,BINANCE:IQUSDT,BINANCE:IRISUSDT,BINANCE:JASMYUSDT,BINANCE:JASMYUSDT.P,BINANCE:JOEUSDT,BINANCE:JOEUSDT.P,BINANCE:JSTUSDT,BINANCE:JTOUSDT,BINANCE:JTOUSDT.P,BINANCE:JUPUSDT,BINANCE:JUPUSDT.P,BINANCE:KASUSDT.P,BINANCE:KAVAUSDT,BINANCE:KAVAUSDT.P,BINANCE:KEYUSDT,BINANCE:KEYUSDT.P,BINANCE:KLAYUSDT,BINANCE:KLAYUSDT.P,BINANCE:KMDUSDT,BINANCE:KNCUSDT,BINANCE:KNCUSDT.P,BINANCE:LDOUSDT,BINANCE:LDOUSDT.P,BINANCE:LEVERUSDT,BINANCE:LEVERUSDT.P,BINANCE:LINAUSDT,BINANCE:LINAUSDT.P,BINANCE:LINKUSDT,BINANCE:LINKUSDT.P,BINANCE:LITUSDT.P,BINANCE:LOKAUSDT,BINANCE:LOOMUSDT,BINANCE:LOOMUSDT.P,BINANCE:LPTUSDT.P,BINANCE:LQTYUSDT,BINANCE:LQTYUSDT.P,BINANCE:LRCUSDT,BINANCE:LRCUSDT.P,BINANCE:LSKUSDT.P,BINANCE:LTOUSDT,BINANCE:LUNA2USDT.P,BINANCE:LUNAUSDT,BINANCE:LUNCUSDT,BINANCE:MAGICUSDT,BINANCE:MAGICUSDT.P,BINANCE:MANAUSDT,BINANCE:MANAUSDT.P,BINANCE:MANTAUSDT,BINANCE:MANTAUSDT.P,BINANCE:MASKUSDT.P,BINANCE:MATICUSDT,BINANCE:MATICUSDT.P,BINANCE:MAVIAUSDT.P,BINANCE:MAVUSDT,BINANCE:MAVUSDT.P,BINANCE:MBLUSDT,BINANCE:MBOXUSDT,BINANCE:MDTUSDT,BINANCE:MDXUSDT,BINANCE:MEMEUSDT,BINANCE:MEMEUSDT.P,BINANCE:MINAUSDT,BINANCE:MINAUSDT.P,BINANCE:MTLUSDT.P,BINANCE:MYROUSDT.P,BINANCE:NEARUSDT,BINANCE:NEARUSDT.P,BINANCE:NFPUSDT,BINANCE:NFPUSDT.P,BINANCE:NKNUSDT,BINANCE:NKNUSDT.P,BINANCE:NOTUSDT,BINANCE:NOTUSDT.P,BINANCE:NTRNUSDT,BINANCE:NTRNUSDT.P,BINANCE:OAXUSDT,BINANCE:OCEANUSDT,BINANCE:OCEANUSDT.P,BINANCE:OGNUSDT,BINANCE:OGNUSDT.P,BINANCE:OMGUSDT,BINANCE:OMGUSDT.P,BINANCE:OMUSDT,BINANCE:OMUSDT.P,BINANCE:ONDOUSDT.P,BINANCE:ONEUSDT,BINANCE:ONEUSDT.P,BINANCE:ONGUSDT,BINANCE:ONGUSDT.P,BINANCE:ONTUSDT,BINANCE:ONTUSDT.P,BINANCE:OOKIUSDT,BINANCE:OPUSDT,BINANCE:OPUSDT.P,BINANCE:ORBSUSDT.P,BINANCE:ORDIUSDT.P,BINANCE:OXTUSDT,BINANCE:OXTUSDT.P,BINANCE:PDAUSDT,BINANCE:PENDLEUSDT,BINANCE:PENDLEUSDT.P,BINANCE:PEOPLEUSDT,BINANCE:PEOPLEUSDT.P,BINANCE:PEPEUSDT,BINANCE:PERPUSDT,BINANCE:PERPUSDT.P,BINANCE:PHAUSDT,BINANCE:PHBUSDT.P,BINANCE:PIXELUSDT,BINANCE:PIXELUSDT.P,BINANCE:POLYXUSDT,BINANCE:POLYXUSDT.P,BINANCE:PONDUSDT,BINANCE:PORTALUSDT,BINANCE:PORTALUSDT.P,BINANCE:POWRUSDT,BINANCE:POWRUSDT.P,BINANCE:PYTHUSDT,BINANCE:PYTHUSDT.P,BINANCE:QIUSDT,BINANCE:QKCUSDT,BINANCE:QUICKUSDT,BINANCE:RADUSDT,BINANCE:RAREUSDT,BINANCE:RDNTUSDT,BINANCE:RDNTUSDT.P,BINANCE:REEFUSDT,BINANCE:REEFUSDT.P,BINANCE:REIUSDT,BINANCE:RENUSDT,BINANCE:RENUSDT.P,BINANCE:REQUSDT,BINANCE:REZUSDT,BINANCE:REZUSDT.P,BINANCE:RIFUSDT,BINANCE:RIFUSDT.P,BINANCE:RNDRUSDT,BINANCE:RNDRUSDT.P,BINANCE:RONINUSDT.P,BINANCE:ROSEUSDT,BINANCE:ROSEUSDT.P,BINANCE:RSRUSDT,BINANCE:RSRUSDT.P,BINANCE:RUNEUSDT,BINANCE:RUNEUSDT.P,BINANCE:RVNUSDT,BINANCE:RVNUSDT.P,BINANCE:SAGAUSDT,BINANCE:SAGAUSDT.P,BINANCE:SANDUSDT,BINANCE:SANDUSDT.P,BINANCE:SCRTUSDT,BINANCE:SCUSDT,BINANCE:SEIUSDT,BINANCE:SEIUSDT.P,BINANCE:SFPUSDT.P,BINANCE:SHIBUSDT,BINANCE:SKLUSDT,BINANCE:SKLUSDT.P,BINANCE:SLPUSDT,BINANCE:SNTUSDT,BINANCE:SNXUSDT,BINANCE:SNXUSDT.P,BINANCE:SOLUSDT.P,BINANCE:SPELLUSDT,BINANCE:SPELLUSDT.P,BINANCE:STEEMUSDT,BINANCE:STEEMUSDT.P,BINANCE:STGUSDT,BINANCE:STGUSDT.P,BINANCE:STMXUSDT,BINANCE:STMXUSDT.P,BINANCE:STORJUSDT,BINANCE:STORJUSDT.P,BINANCE:STPTUSDT,BINANCE:STRAXUSDT,BINANCE:STRKUSDT,BINANCE:STRKUSDT.P,BINANCE:STXUSDT,BINANCE:STXUSDT.P,BINANCE:SUIUSDT,BINANCE:SUIUSDT.P,BINANCE:SUNUSDT,BINANCE:SUPERUSDT.P,BINANCE:SUSHIUSDT,BINANCE:SUSHIUSDT.P,BINANCE:SXPUSDT,BINANCE:SXPUSDT.P,BINANCE:SYNUSDT,BINANCE:SYSUSDT,BINANCE:TFUELUSDT,BINANCE:THETAUSDT,BINANCE:THETAUSDT.P,BINANCE:TIAUSDT.P,BINANCE:TLMUSDT,BINANCE:TLMUSDT.P,BINANCE:TNSRUSDT,BINANCE:TNSRUSDT.P,BINANCE:TOKENUSDT.P,BINANCE:TONUSDT.P,BINANCE:TROYUSDT,BINANCE:TRUUSDT,BINANCE:TRUUSDT.P,BINANCE:TRXUSDT,BINANCE:TRXUSDT.P,BINANCE:TUSDT,BINANCE:TUSDT.P,BINANCE:TWTUSDT.P,BINANCE:UMAUSDT.P,BINANCE:UNFIUSDT.P,BINANCE:UNIUSDT,BINANCE:UNIUSDT.P,BINANCE:USDCUSDT,BINANCE:USDTBRL,BINANCE:USDTDAI,BINANCE:USDTTRY,BINANCE:USTCUSDT,BINANCE:USTCUSDT.P,BINANCE:UTKUSDT,BINANCE:VANRYUSDT,BINANCE:VANRYUSDT.P,BINANCE:VETUSDT,BINANCE:VETUSDT.P,BINANCE:VGXUSDT,BINANCE:VIBUSDT,BINANCE:VICUSDT,BINANCE:VIDTUSDT,BINANCE:VITEUSDT,BINANCE:VOXELUSDT,BINANCE:VTHOUSDT,BINANCE:WANUSDT,BINANCE:WAVESUSDT.P,BINANCE:WAXPUSDT,BINANCE:WAXPUSDT.P,BINANCE:WIFUSDT,BINANCE:WIFUSDT.P,BINANCE:WINUSDT,BINANCE:WLDUSDT,BINANCE:WLDUSDT.P,BINANCE:WOOUSDT,BINANCE:WOOUSDT.P,BINANCE:WRXUSDT,BINANCE:WUSDT,BINANCE:WUSDT.P,BINANCE:XAIUSDT,BINANCE:XAIUSDT.P,BINANCE:XECUSDT,BINANCE:XEMUSDT,BINANCE:XEMUSDT.P,BINANCE:XLMUSDT,BINANCE:XLMUSDT.P,BINANCE:XRPUSDT,BINANCE:XRPUSDT.P,BINANCE:XTZUSDT,BINANCE:XTZUSDT.P,BINANCE:XVGUSDT,BINANCE:XVGUSDT.P,BINANCE:YGGUSDT,BINANCE:YGGUSDT.P,BINANCE:ZETAUSDT.P,BINANCE:ZILUSDT,BINANCE:ZILUSDT.P,BINANCE:ZRXUSDT,BINANCE:ZRXUSDT.P,BINANCE:AVAXUSDT"
        old_pairs_list = old_pairs_str.replace("BINANCE:", "").split(",")
        old_pair = [pair for pair in old_pairs_list if ".P" not in pair]

        symbols = set(all_pairs) - set(old_pair)

        # Example usage
        timeframe = '1m'  # Replace with your desired timeframe
        from_date = '2017-01-01'  # Replace with your start date
        to_date = '2024-06-05'  # Replace with your end date
        batch_size = 1000  # Adjust batch size if needed

        main(symbols, timeframe, from_date, to_date, batch_size)




    except Exception as e:
        print(f"Error in __main__ execution: {e}")
