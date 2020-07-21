import pandas as pd
import numpy as np
from joblib import dump, load
import crypto_stream

MODEL = load('random_forest_model_1.joblib')
#STATERGIES = ['crossover_signal', 'vol_trend_signal', 'bollinger_signal']


def load_model():
    #return load('random_forest_model_1.joblib')
    return MODEL
    
def predict(df_ee, no_of_data=22):
    future_df = crypto_stream.get_data_from_table(no_of_data)
    print(len(future_df))
    if len(future_df)<=no_of_data:
        return df_ee
    future_df = get_trading_singals(future_df)
    future_predict = future_df.tail(2)[get_statergies]
    predictions = MODEL.predict(future_predict)
    entry_exit = predictions[1]-predictions[0]
    if df_ee is None:
        df_ee = future_df.iloc[[-1],:1]
    else:
        df_ee.append(future_df.iloc[[-1],:1])
    df_ee['entry/exit'][-1]=entry_exit
    print('-----------------')
    print(df_ee)
    return df_ee
    

def get_statergies():
    return ['crossover_signal', 'vol_trend_signal', 'bollinger_signal']
    
def get_trading_singals(stock_df):
    # Drop NAs and calculate daily percent return
    stock_df['daily_return'] = stock_df['close'].dropna().pct_change()


    # Set short and long windows
    short_window = 1
    long_window = 10

    # Construct a `Fast` and `Slow` Exponential Moving Average from short and long windows, respectively
    stock_df['fast_close'] = stock_df['close'].ewm(halflife=short_window).mean()
    stock_df['slow_close'] = stock_df['close'].ewm(halflife=long_window).mean()

    # Construct a crossover trading signal
    stock_df['crossover_long'] = np.where(stock_df['fast_close'] > stock_df['slow_close'], 1.0, 0.0)
    stock_df['crossover_short'] = np.where(stock_df['fast_close'] < stock_df['slow_close'], -1.0, 0.0)
    stock_df['crossover_signal'] = stock_df['crossover_long'] + stock_df['crossover_short']


    # Plot the EMA of BTC/USD closing prices
    stock_df[['close', 'fast_close', 'slow_close']].plot(figsize=(20,10))


    # Set short and long volatility windows
    short_vol_window = 1
    long_vol_window = 10

    # Construct a `Fast` and `Slow` Exponential Moving Average from short and long windows, respectively
    stock_df['fast_vol'] = stock_df['daily_return'].ewm(halflife=short_vol_window).std()
    stock_df['slow_vol'] = stock_df['daily_return'].ewm(halflife=long_vol_window).std()

    # Construct a crossover trading signal
    stock_df['vol_trend_long'] = np.where(stock_df['fast_vol'] < stock_df['slow_vol'], 1.0, 0.0)
    stock_df['vol_trend_short'] = np.where(stock_df['fast_vol'] > stock_df['slow_vol'], -1.0, 0.0) 
    stock_df['vol_trend_signal'] = stock_df['vol_trend_long'] + stock_df['vol_trend_short']


    # Plot the EMA of BTC/USD daily return volatility
    stock_df[['fast_vol', 'slow_vol']].plot(figsize=(20,10))

    # Set bollinger band window
    bollinger_window = 20

    # Calculate rolling mean and standard deviation
    stock_df['bollinger_mid_band'] = stock_df['close'].rolling(window=bollinger_window).mean()
    stock_df['bollinger_std'] = stock_df['close'].rolling(window=20).std()

    # Calculate upper and lowers bands of bollinger band
    stock_df['bollinger_upper_band']  = stock_df['bollinger_mid_band'] + (stock_df['bollinger_std'] * 1)
    stock_df['bollinger_lower_band']  = stock_df['bollinger_mid_band'] - (stock_df['bollinger_std'] * 1)

    # Calculate bollinger band trading signal
    stock_df['bollinger_long'] = np.where(stock_df['close'] < stock_df['bollinger_lower_band'], 1.0, 0.0)
    stock_df['bollinger_short'] = np.where(stock_df['close'] > stock_df['bollinger_upper_band'], -1.0, 0.0)
    stock_df['bollinger_signal'] = stock_df['bollinger_long'] + stock_df['bollinger_short']

    # Plot the Bollinger Bands for BTC/USD closing prices
    stock_df[['close','bollinger_mid_band','bollinger_upper_band','bollinger_lower_band']].plot(figsize=(20,10))

    return stock_df