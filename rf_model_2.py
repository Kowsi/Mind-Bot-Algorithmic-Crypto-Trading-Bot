import pandas as pd
import numpy as np
from joblib import dump, load
import crypto_stream
import talib as ta
#MODEL = load('random_forest_model_2.joblib')
#STATERGIES = ['crossover_sma_signal', 'crossover_ema_signal', 'bollinger_signal', 'rsi_signal', 'stoch_signal', 'macd_signal']


def load_model(model_name):
    return load('random_forest_model_2.joblib')
    #return MODEL
    
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
    return ['crossover_sma_signal', 'crossover_ema_signal', 'bollinger_signal', 'rsi_signal', 'stoch_signal', 'macd_signal']
     
    
def get_trading_singals(stock_df):
    # Drop NAs and calculate daily percent return
    stock_df['daily_return'] = stock_df['close'].dropna().pct_change()
    # Construct a `Fast` and `Slow` Simple Moving Average 
    stock_df['SMA 50'] = ta.SMA(stock_df['close'],50)
    stock_df['SMA 100'] = ta.SMA(stock_df['close'], 100)
    # Construct a crossover trading signal - SMA
    stock_df['crossover_sma_buy'] = np.where(stock_df['SMA 50'] > stock_df['SMA 100'], 1.0, 0.0)
    stock_df['crossover_sma_sell'] = np.where(stock_df['SMA 50'] < stock_df['SMA 100'], -1.0, 0.0)
    stock_df['crossover_sma_signal'] = stock_df['crossover_sma_buy'] + stock_df['crossover_sma_sell']
    # Construct a `Fast` and `Slow` Exponential Moving Average 

    stock_df['EMA 50'] = ta.EMA(stock_df['close'], timeperiod = 50)
    stock_df['EMA 100'] = ta.EMA(stock_df['close'], timeperiod = 100)
    # Construct a crossover trading signal EMA
    stock_df['crossover_ema_buy'] = np.where(stock_df['EMA 50'] > stock_df['EMA 100'], 1.0, 0.0)
    stock_df['crossover_ema_sell'] = np.where(stock_df['EMA 50'] < stock_df['EMA 100'], -1.0, 0.0)
    stock_df['crossover_ema_signal'] = stock_df['crossover_ema_buy'] + stock_df['crossover_ema_sell']
    # Bollinger Bands


    stock_df['upper_band'], stock_df['middle_band'], stock_df['lower_band'] = ta.BBANDS(stock_df['close'], timeperiod =20)
    # Calculate bollinger band trading signal
    stock_df['bollinger_buy'] = np.where(stock_df['close'] < stock_df['lower_band'], 1.0, 0.0)
    stock_df['bollinger_sell'] = np.where(stock_df['close'] > stock_df['upper_band'], -1.0, 0.0)
    stock_df['bollinger_signal'] = stock_df['bollinger_buy'] + stock_df['bollinger_sell']
    # Relative Strength Index
    stock_df['RSI'] = ta.RSI(stock_df['close'],20)
    # Calculate RSI trading signal
    lower = 30
    upper = 70
    stock_df['rsi_buy'] = np.where(stock_df['RSI'] < lower, 1.0, 0.0)
    stock_df['rsi_sell'] = np.where(stock_df['RSI'] > upper, -1.0, 0.0)
    stock_df['rsi_signal'] = stock_df['rsi_buy'] + stock_df['rsi_sell']
    # Stochastic Oscillator
    stock_df['slowk'], stock_df['slowd'] = ta.STOCH(stock_df['high'], stock_df['low'], stock_df['close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    # Generating Stochastic Oscillator Trading Signals
    stoch_lower = 20
    stoch_upper = 80
    stock_df['stoch_Buy'] = np.where((stock_df['slowk'] > stock_df['slowd']) & (stock_df['slowk'].shift(1) < stock_df['slowd'].shift(1)) & (stock_df['slowd'] < stoch_lower), 1.0, 0.0)
    stock_df['stoch_Sell'] = np.where((stock_df['slowk'] < stock_df['slowd']) & (stock_df['slowk'].shift(1) > stock_df['slowd'].shift(1)) & (stock_df['slowd'] > stoch_upper), 1.0, 0.0)
    stock_df['stoch_signal'] = stock_df['stoch_Buy'] + stock_df['stoch_Sell']
    # Moving Average Convergence/Divergence
    import math
    macd, signal, hist = ta.MACD(stock_df['close'], fastperiod=14, slowperiod=30, signalperiod=11)
    stock_df['norm_macd'] = np.nan_to_num(macd) / math.sqrt(np.var(np.nan_to_num(macd)))
    stock_df['norm_signal'] = np.nan_to_num(signal) / math.sqrt(np.var(np.nan_to_num(signal)))
    stock_df['norm_hist'] = np.nan_to_num(hist) / math.sqrt(np.var(np.nan_to_num(hist)))
    #stock_df['macdrocp'] = ta.ROCP(stock_df['norm_macd'] + np.max(stock_df['norm_macd']) - np.min(stock_df['norm_macd']), timeperiod=1)
    #stock_df['signalrocp'] = ta.ROCP(stock_df['norm_signal'] + np.max(stock_df['norm_signal']) - np.min(stock_df['norm_signal']), timeperiod=1)
    #stock_df['histrocp'] = ta.ROCP(stock_df['norm_hist'] + np.max(stock_df['norm_hist']) - np.min(stock_df['norm_hist']), timeperiod=1)
    # Generating MACD Signals
    stock_df['macd_buy'] = np.where(stock_df['norm_signal'] < stock_df['norm_macd'], 1.0, 0.0)
    stock_df['macd_sell'] = np.where(stock_df['norm_signal'] > stock_df['norm_macd'], -1.0, 0.0)
    stock_df['macd_signal'] = stock_df['macd_buy'] + stock_df['macd_sell']
    return stock_df