import pandas as pd
import numpy as np
import crypto_stream
import rf_model
import rf_model_2
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

MODEL_LIST = ['Random Forest Classifier - 1', 'Random Forest Classifier - 2']


def test():
    portfolio_metrics, trade_metrics, portfolio_evaluation = main('BTC/USD', MODEL_LIST[0] ,'1m')
    return portfolio_metrics, trade_metrics, portfolio_evaluation
    
def main(crypto, model_name, timeframe, inital_capital=100000.0, no_of_shares=10):
    trading_signals_df, results, acc_score = get_prediction(crypto, model_name, timeframe)
    signals_df = concat_prediction_signals(trading_signals_df, results)
    portfolio_metrics = calculate_portfolio_metrics(signals_df, inital_capital, no_of_shares)
    trade_metrics = calculate_trade_metrics(crypto, portfolio_metrics)
    portfolio_evaluation = evaluate_portfolio_metrics(portfolio_metrics, trade_metrics)
    return portfolio_metrics, trade_metrics, portfolio_evaluation

def concat_prediction_signals(trading_signals_df, results):
    results['Entry/Exit'] = results['Predicted Value'].diff()
    results.dropna(inplace=True)
    results.columns = ['Positive Return', 'Signal', 'Entry/Exit']
    
    signals_df = pd.concat([results, trading_signals_df], join='inner', axis=1)

    return signals_df

def model_list():
    return MODEL_LIST

def get_model(model_name):
    if(model_name=='Random Forest Classifier - 1'):
        return rf_model
    if(model_name=='Random Forest Classifier - 2'):
        return rf_model_2
    return rf_model
    
def get_prediction(crypto, model_name, timeframe):
    package = get_model(model_name)
    model = package.load_model(model_name)
    hist_data = crypto_stream.fetch_historical_data(crypto=crypto, interval=timeframe, limit=720)
    print(f'total_data---{len(hist_data)}')
    trading_signals_df = package.get_trading_singals(hist_data)
    X_test, y_test = format_test_data(trading_signals_df, package)
    results, acc_score = model_predict_test_data(model, X_test, y_test)
    return trading_signals_df, results, acc_score
    
    
def format_test_data(trading_signals_df, package):
    # Set x variable list of features
    x_var_list = package.get_statergies()

    # Filter by x-variable list
    trading_signals_df[x_var_list].tail()

    # Shift DataFrame values by 1
    trading_signals_df[x_var_list] = trading_signals_df[x_var_list].shift(1)

    # Drop NAs and replace positive/negative infinity values
    trading_signals_df.dropna(subset=x_var_list, inplace=True)
    trading_signals_df.dropna(subset=['daily_return'], inplace=True)
    trading_signals_df = trading_signals_df.replace([np.inf, -np.inf], np.nan)

    # Construct the dependent variable where if daily return is greater than 0, then 1, else, 0.
    trading_signals_df['Positive Return'] = np.where(trading_signals_df['daily_return'] > 0, 1.0, 0.0)

    # Construct the X test and y test datasets
    X_test = trading_signals_df[x_var_list]
    y_test = trading_signals_df['Positive Return']
    return X_test, y_test


def model_predict_test_data(model, X_test, y_test):
    # Make a prediction of "y" values from the X_test dataset
    predictions = model.predict(X_test)

    # Assemble actual y data (Y_test) with predicted y data (from just above) into two columns in a dataframe:
    Results = y_test.to_frame()
    Results["Predicted Value"] = predictions
    
    # Calculating the accuracy score
    acc_score = accuracy_score(y_test, predictions)
    return Results, acc_score



def calculate_portfolio_metrics(signals_df, inital_capital, share_size):
    # Set initial capital
    initial_capital = float(inital_capital)
    # Set the share size
    #share_size = 10
    
    # If predicted signals starts with sell, slice the dataset
    signal_start = signals_df[signals_df['Entry/Exit'].isin([-1,1])]
    if(signal_start.iloc[0]['Entry/Exit']==-1):
        signals_df = signals_df.iloc[signal_start.iloc[[0]].index[0]<signals_df.index]

    # Take a 500 share position where the dual moving average crossover is 1 (SMA50 is greater than SMA100)
    signals_df['Position'] = share_size * signals_df['Signal']
    # Find the points in time where a 500 share position is bought or sold
    signals_df['Entry/Exit Position'] = signals_df['Entry/Exit'] * share_size
    # Multiply share price by entry/exit positions and get the cumulatively sum
    signals_df['Portfolio Holdings'] = signals_df['close'] * signals_df['Entry/Exit Position'].cumsum()
    # Subtract the initial capital by the portfolio holdings to get the amount of liquid cash in the portfolio
    signals_df['Portfolio Cash'] = initial_capital - (signals_df['close'] * signals_df['Entry/Exit Position']).cumsum()
    # Get the total portfolio value by adding the cash amount by the portfolio holdings (or investments)
    signals_df['Portfolio Total'] = signals_df['Portfolio Cash'] + signals_df['Portfolio Holdings']
    # Calculate the portfolio daily returns
    signals_df['Portfolio Daily Returns'] = signals_df['Portfolio Total'].pct_change()
    # Calculate the cumulative returns
    signals_df['Portfolio Cumulative Returns'] = (1 + signals_df['Portfolio Daily Returns']).cumprod() - 1

    return signals_df


def calculate_trade_metrics(crypto, signals_df):
    trade_evaluation_df = pd.DataFrame(
    columns=[
        'Stock', 
        'Entry Date', 
        'Exit Date', 
        'Shares', 
        'Entry Share Price', 
        'Exit Share Price', 
        'Entry Portfolio Holding', 
        'Exit Portfolio Holding', 
        'Profit/Loss'])
    # Initialize iterative variables
    entry_date = ''
    exit_date = ''
    entry_portfolio_holding = 0
    exit_portfolio_holding = 0
    share_size = 0
    entry_share_price = 0
    exit_share_price = 0
    # Loop through signal DataFrame
    # If `Entry/Exit` is 1, set entry trade metrics
    # Else if `Entry/Exit` is -1, set exit trade metrics and calculate profit,
    # Then append the record to the trade evaluation DataFrame
    for index, row in signals_df.iterrows():
        if row['Entry/Exit'] == 1:
            entry_date = index
            entry_portfolio_holding = abs(row['Portfolio Holdings'])
            share_size = row['Entry/Exit Position']
            entry_share_price = row['close']
        elif row['Entry/Exit'] == -1:
            exit_date = index
            exit_portfolio_holding = abs(row['close'] * row['Entry/Exit Position'])
            exit_share_price = row['close']
            profit_loss =  exit_portfolio_holding - entry_portfolio_holding 
            trade_evaluation_df = trade_evaluation_df.append(
                {
                    'Stock': crypto,
                    'Entry Date': entry_date,
                    'Exit Date': exit_date,
                    'Shares': share_size,
                    'Entry Share Price': entry_share_price,
                    'Exit Share Price': exit_share_price,
                    'Entry Portfolio Holding': entry_portfolio_holding,
                    'Exit Portfolio Holding': exit_portfolio_holding,
                    'Profit/Loss': profit_loss
                },
                ignore_index=True)
    print(trade_evaluation_df['Profit/Loss'].sum())
    return trade_evaluation_df


def evaluate_portfolio_metrics(signals_df, trade_evaluation_df):
    # Prepare DataFrame for metrics
    metrics = [
        'Annual Return',
        'Cumulative Returns',
        'Annual Volatility',
        'Sharpe Ratio',
        'Sortino Ratio',
        'Total Profit/Loss',
        'Test Dataset Size',
    ]

    columns = ['Backtest']

    # Initialize the DataFrame with index set to evaluation metrics and column as `Backtest` (just like PyFolio)
    portfolio_evaluation_df = pd.DataFrame(index=metrics, columns=columns).rename_axis('Metrics')
    portfolio_evaluation_df
    
    # Calculate cumulative return

    portfolio_evaluation_df.loc['Cumulative Returns'] = (signals_df['Portfolio Cumulative Returns'][-1])
    
    # Calculate annualized return
    portfolio_evaluation_df.loc['Annual Return'] = (
        signals_df['Portfolio Daily Returns'].mean() * 252
    )

    # Calculate annual volatility
    portfolio_evaluation_df.loc['Annual Volatility'] = (
        signals_df['Portfolio Daily Returns'].std() * np.sqrt(252)
    )

    # Calculate Sharpe Ratio
    portfolio_evaluation_df.loc['Sharpe Ratio'] = (
        signals_df['Portfolio Daily Returns'].mean() * 252) / (
        signals_df['Portfolio Daily Returns'].std() * np.sqrt(252)
    )

    # Calculate Downside Return
    sortino_ratio_df = signals_df[['Portfolio Daily Returns']].copy()
    sortino_ratio_df.loc[:,'Downside Returns'] = 0

    target = 0
    mask = sortino_ratio_df['Portfolio Daily Returns'] < target
    sortino_ratio_df.loc[mask, 'Downside Returns'] = sortino_ratio_df['Portfolio Daily Returns']**2
    portfolio_evaluation_df

    # Calculate Sortino Ratio
    down_stdev = np.sqrt(sortino_ratio_df['Downside Returns'].mean()) * np.sqrt(252)
    expected_return = sortino_ratio_df['Portfolio Daily Returns'].mean() * 252
    sortino_ratio = expected_return/down_stdev

    portfolio_evaluation_df.loc['Sortino Ratio'] = sortino_ratio
    
    portfolio_evaluation_df.loc['Total Profit/Loss'] = trade_evaluation_df['Profit/Loss'].sum()
    
    portfolio_evaluation_df.loc['Test Dataset Size'] = signals_df.shape[0]
    
    portfolio_evaluation_df
    return portfolio_evaluation_df.round(2)


