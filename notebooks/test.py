import ccxt
import asyncio
import datetime

# Create an instance of the Binance exchange
exchange = ccxt.binance()

# Set the symbol to BTC/USDT
symbol = 'BTC/USDT'

# Set the timeframe to 1 minute
timeframe = '1m'

# Set the start and end dates
start_date = '2022-01-01T00:00:00Z'
end_date = '2022-01-31T23:59:59Z'

# Set the number of candles to fetch per chunk
limit = 1000

# Convert start and end dates to timestamps
start_timestamp = exchange.parse8601(start_date)
end_timestamp = exchange.parse8601(end_date)

# Define a function to fetch OHLCV data for a chunk
def fetch_ohlcv_chunk(symbol, timeframe, start_timestamp, limit):
    return exchange.fetch_ohlcv(symbol, timeframe, since=start_timestamp, limit=limit)

# Define an async function to fetch OHLCV data in chunks
async def fetch_ohlcv_data():
    global start_timestamp
    while start_timestamp < end_timestamp:
        # Fetch OHLCV data for the current chunk
        ohlcv = await asyncio.to_thread(fetch_ohlcv_chunk, symbol, timeframe, start_timestamp, limit)

        # Print the OHLCV data for the current chunk
        for candle in ohlcv:
            print(candle)

        # Update the start timestamp for the next chunk
        if len(ohlcv) < limit:
            break
        start_timestamp = ohlcv[-1][0] + 1  # Increment timestamp to avoid duplicate data

# Run the async function
asyncio.run(fetch_ohlcv_data())
