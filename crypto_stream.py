import sqlite3
import pandas as pd
import os
import time
import pickle
import ccxt
import numpy as np
import datetime

# it's ok to use one shared sqlite connection
# as we are making selects only, no need for any kind of serialization as well
#Global Variables
kraken_public_key = os.getenv("KRAKEN_PUBLIC_KEY")
kraken_secret_key = os.getenv("KRAKEN_SECRET_KEY")
SQLITE_DATABASE = r"algo_trader_history.sqlite"
HISTORY_TABLE = 'hist_data'
SYMBOL = 'BTC/USD'
CONNECTION = sqlite3.connect("algo_trader_history.sqlite" , check_same_thread=False)
MAX_WINDOW = 100

def init_connection(db_file=SQLITE_DATABASE):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    sql_create_crypto_table = """ CREATE TABLE IF NOT EXISTS hist_data ('date' TIMESTAMP PRIMARY KEY, open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL, volume REAL); """
    conn = None
    try:
        # Initialize Database
        conn = CONNECTION
        #sqlite3.connect(SQLITE_DATABASE, isolation_level=None, check_same_thread=False)
        with conn:
            cur = conn.cursor()
            cur.execute(f"DROP TABLE IF EXISTS {HISTORY_TABLE}")
            cur.execute(sql_create_crypto_table)
            print('cleared data..')
            '''if crypto:
                df = fetch_past_data(crypto)
                df.to_sql(HISTORY_TABLE, CONNECTION, if_exists="replace", index=True, index_label='date')'''
    except sqlite3.Error as e:
        print(e)
    return conn

def get_connection():
    return CONNECTION

def get_table_data(table_name, limit):
    """ Get table data
    :param table_name: table of the table
    :param limit: number of data to be retrieved
    :return:
    """
    try:
        # create a database connection
        conn = get_connection()
        # create tables
        with conn:
            df = pd.read_sql(f"select * from {table_name} limit {limit}", conn, index_col='date')
        return df
    except sqlite3.Error as e:
        print(e)
    return None

def get_data_from_table(max_window=MAX_WINDOW):
    df = pd.read_sql(f"select * from {HISTORY_TABLE} ORDER BY date DESC limit {max_window}", CONNECTION, index_col='date', parse_dates='date')
    return df

# Fetch data from Kraken
def fetch_data(crypto=SYMBOL, max_window=MAX_WINDOW):
    """Fetches the latest prices."""
    #db = sqlite3.connect("algo_trader_history.sqlite" )
    kraken = ccxt.kraken({"apiKey": kraken_public_key, "secret": kraken_secret_key})
    data = kraken.fetch_ticker(crypto)
    data = [[data['timestamp'], data["open"], data["high"] ,data["low"], data["close"], data["baseVolume"]]]
    df = get_dataframe(data)
    try:
        df.to_sql(HISTORY_TABLE, CONNECTION, if_exists="append", index=True)
    except sqlite3.IntegrityError:
        pass
    df = pd.read_sql(f"select * from {HISTORY_TABLE} ORDER BY date DESC limit {max_window}", CONNECTION, index_col='date', parse_dates='date')
    return df


def fetch_historical_data(crypto=SYMBOL, interval='1m', limit=720):
    kraken = ccxt.kraken()
    interval_in_min = {'1m':1,'5m':5, '30m':30, '1h':60, '1d':1440, '1w':10080}
    no_of_data = limit * interval_in_min[interval]
    print(no_of_data)
    past_datetime = (datetime.datetime.now() + datetime.timedelta(minutes=240-no_of_data)).strftime('%Y-%m-%d %H:%M:%S')
    data = kraken.fetch_ohlcv(crypto, interval, kraken.parse8601(past_datetime))
    time.sleep(1)
    return get_dataframe(data)

def fetch_past_data(crypto=SYMBOL, interval='1m', limit=30):
    data = ccxt.kraken().fetch_ohlcv(crypto, interval, limit)
    time.sleep(1)
    return get_dataframe(data)

def get_dataframe(data):
    df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df.date,unit='ms')
    df.index=pd.DatetimeIndex(df.date).tz_localize('UTC').tz_convert('US/Eastern')
    df.drop('date', axis=1,inplace=True)
    return df
    
def generate_signals(df):
    #df = get_table_data(table_name, limit)
    """Generates trading signals for a given dataset."""
    
    # Set window
    short_window = 10

    signals = df.iloc[::-1].copy()
    signals["signal"] = 0.0

    # Generate the short and long moving averages
    signals["sma10"] = signals["close"].rolling(window=10).mean()
    signals["sma20"] = signals["close"].rolling(window=20).mean()

    # Generate the trading signal 0 or 1,
    signals["signal"] = np.where(
        signals["sma10"] > signals["sma20"], 1.0, 0.0
    )
    signals.loc[:short_window-1, ["signal"]] = 0.0
    # Calculate the points in time at which a position should be taken, 1 or -1
    signals["entry/exit"] = signals["signal"].diff()
    #print(signals)
    return signals


def execute_trade_strategy(signals, account):
    """Makes a buy/sell/hold decision."""
    if signals["entry/exit"][-1] == 1.0:
        account['status'] = "buy"
        account['close']=signals["close"][-1]
        number_to_buy = round(account["balance"] / signals["close"][-1], 0) * 0.001
        account["balance"] -= number_to_buy * signals["close"][-1]
        account["shares"] += number_to_buy
        
    elif signals["entry/exit"][-1] == -1.0:
        account['status'] = "sell"
        account['close']=signals["close"][-1]
        account["balance"] += signals["close"][-1] * account["shares"]
        account["shares"] = 0
    else:
        return None
    
    return account



def get_crypto_symbols():
    exchange = ccxt.kraken()
    exchange.load_markets()
    return [ele for ele in exchange.symbols if ele.endswith("/USD")]

#init_connection()