from lightweight_charts import Chart
from pathlib import Path
import pandas as pd

def on_timeframe_selection(chart):
    print(f'Getting data with a {chart.topbar["menu"].value} timeframe.')
    
    
if __name__ == '__main__':
    cwd = Path.cwd()
    data_path = cwd / 'data' / 'clean' / 'btcusdt.parquet'
    df = pd.read_parquet(data_path)
    df = df.iloc[-5000:]
    
    # chart = Chart()
    # chart.set(df)

    chart = Chart(inner_width=0.7, inner_height=1, toolbox=True)
    # subchart = chart.create_subchart(width=0.3, height=0.5)
    chart.set(df)
    # subchart.set(df)

    chart.topbar.switcher(
        name='my_switcher',
        options=('1min', '5min', '30min'),
        default='5min',
        align='middle',
        func=on_timeframe_selection)
    
    chart.topbar.menu('menu', 
                      options=('1min', '5min', '30min'),
                      func=on_timeframe_selection,
                      default="1min",
                      separator=True,
                      align='right')
    
    table = chart.create_table(width=0.3, height=1,
                    headings=('Ticker', 'Quantity', 'Status', '%', 'PL'),
                    widths=(0.2, 0.1, 0.2, 0.2, 0.3),
                    alignments=('center', 'center', 'right', 'right', 'right'),
                    position='right', func=on_timeframe_selection)

    table.format('PL', f'£ {table.VALUE}')
    table.format('%', f'{table.VALUE} %')

    table.new_row('SPY', 3, 'Submitted', 0, 0)
    table.new_row('AMD', 1, 'Filled', 25.5, 105.24)
    table.new_row('NVDA', 2, 'Filled', -0.5, -8.24)

    table.footer(2)
    table.footer[0] = 'Selected:'

    chart.show(block=True)
    # menu(name: str, options: tuple: default: str, separator: bool, align: ALIGN, func: callable)