import os
import pathlib
import json
import base64
import datetime
import requests
import pathlib
import math
import pandas as pd
import flask

import plotly.graph_objs as go
from plotly import tools

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table
import dash_daq as daq
import ccxt
import crypto_stream
from dash.exceptions import PreventUpdate
import rf_model
import time
import backtesting
import dash_table.FormatTemplate as FormatTemplate
crypto_stream.init_connection()

server = flask.Flask(__name__)

app = dash.Dash(
    __name__,
    server=server,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

app.config["suppress_callback_exceptions"] = True

APP_PATH = str(pathlib.Path(__file__).parent.resolve())
STREAM_TABLE = dict(id='stream-table',data=[{'close':0.0, 
                   "balance": 10000, 
                   "shares": 0, 
                   'status':''}
                 ],
            columns=[{'id':'close', 'name':'Close'},
                     {'id':'balance', 'name':'Balance'},
                     {'id':'shares', 'name':'Shares'},
                     {'id':'status', 'name':'Status'}])
columns=[
        'Stock', 
        'Entry Date', 
        'Exit Date', 
        'Shares', 
        'Entry Share Price', 
        'Exit Share Price', 
        'Entry Portfolio Holding', 
        'Exit Portfolio Holding', 
        'Profit/Loss']
#dict(id='trade-metric-table',data=[],columns=[{'id':col, 'name':col} for col in columns])
TRADE_METRIC_TABLE = dict(id='trade-metric-table', data=[], 
                          columns=[{'id':'Stock', 'name':'Stock'},
                     {'id':'Entry Date', 'name':'Entry Date', 'type': 'datetime'},
                     {'id':'Exit Date', 'name':'Exit Date', 'type': 'datetime'},
                     {'id':'Shares', 'name':'Shares'},
                     {'id':'Entry Share Price', 'name':'Entry Share Price', 'type':'numeric','format': FormatTemplate.money(2)},
                     {'id':'Exit Share Price', 'name':'Exit Share Price','type':'numeric','format': FormatTemplate.money(2)},
                     {'id':'Entry Portfolio Holding', 'name':'Entry Portfolio Holding', 'type':'numeric','format': FormatTemplate.money(2)},
                     {'id':'Exit Portfolio Holding', 'name':'Exit Portfolio Holding', 'type':'numeric','format': FormatTemplate.money(2)},
                     {'id':'Profit/Loss', 'name':'Profit/Loss', 'type':'numeric','format': FormatTemplate.money(2)}])


# API Requests for news div
news_requests = requests.get(
    "https://newsapi.org/v2/top-headlines?sources=bbc-news&apiKey=da8e2e705b914f9f86ed2e9692e66012"
)

# API Call to update news
def update_news():
    json_data = news_requests.json()["articles"]
    df = pd.DataFrame(json_data)
    df = pd.DataFrame(df[["title", "url"]])
    max_rows = 10
    return html.Div(
        children=[
            html.P(className="p-news", children="Headlines"),
            html.P(
                className="p-news float-right",
                children="Last update : "
                + datetime.datetime.now().strftime("%H:%M:%S"),
            ),
            html.Table(
                className="table-news",
                children=[
                    html.Tr(
                        children=[
                            html.Td(
                                children=[
                                    html.A(
                                        className="td-link",
                                        children=df.iloc[i]["title"],
                                        href=df.iloc[i]["url"],
                                        target="_blank",
                                    )
                                ]
                            )
                        ]
                    )
                    for i in range(min(len(df), max_rows))
                ],
            ),
        ]
    )

# MAIN CHART TRACES (STYLE tab)
def line_trace(df, y_col, color='rgb(244, 212, 77)'):
    trace = go.Scatter(
        x=df.index, 
        y=df[y_col], 
        mode="lines", 
        showlegend=False, 
        name=y_col,
        line=dict(color=color)
    )
    return trace

def marker_trace(x_data, y_data, symbol, color, name, marker_size=15):
    trace = go.Scatter(
        x=x_data, 
        y=y_data, 
        mode="markers", 
        showlegend=False, 
        marker_size=marker_size,
        marker_symbol=symbol,
        marker_color=color,
        name=name
    )
    return trace

def bar_trace(df, y_col):
    return go.Ohlc(
        x=df.index,
        open=df[y_col],
        increasing=dict(line=dict(color="#888888")),
        decreasing=dict(line=dict(color="#888888")),
        showlegend=False,
        name="bar",
    )
def colored_bar_trace(df):
    return go.Ohlc(
        x=df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        showlegend=False,
        name="colored bar",
    )

def candlestick_trace(df, col):
    return go.Candlestick(
        x=df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        increasing=dict(line=dict(color="#00ff00")),
        decreasing=dict(line=dict(color="white")),
        showlegend=False,
        name="candlestick",
    )

def get_fig_layout(tickformat="%H:%M:%S"):
    layout = dict(margin=dict(t=40),
        hovermode="closest",
        #uirevision=True,
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"font": {"color": "darkgray"}, "orientation": "h", "x": 0, "y": 1.1},
        font={"color": "darkgray"},
        showlegend=True,
        xaxis={
            "zeroline": False,
            "showgrid": False,
            "title": "Closing Price",
            "showline": False,
            #"domain": [0, 0.8],
            "tickformat" : tickformat,
            "titlefont": {"color": "darkgray"},
        },
        yaxis={
            "title": 'Time',
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "autorange": True,
            "titlefont": {"color": "darkgray"},
        },xaxis2={
            "title": "Time",
            #"domain": [0.8, 1],  # 70 to 100 % of width
            "titlefont": {"color": "darkgray"},
            "showgrid": False,
        },
        yaxis2={
            "anchor": "free",
            "overlaying": "y",
            "side": "right",
            "showticklabels": False,
            "titlefont": {"color": "darkgray"},
        },
    )
    return layout
        
def generate_section_banner(title):
    return html.Div(className="section-banner", children=title)

def get_close_fig(df):
    # Add main trace (style) to figure
    '''fig = make_subplots(
        rows=1,
        shared_xaxes=True,
        shared_yaxes=True,
        cols=1,
        print_grid=False,
        vertical_spacing=0.12,
    )
    fig.append_trace(line_trace(df), 1, 1)
    fig.append_trace(bar_trace(df), 2, 1)'''
    fig = go.Figure()
    fig.add_traces([line_trace(df, 'close')])
    fig["layout"] = get_fig_layout()
    return fig
    

def get_sma_fig(df):
    fig = go.Figure()
    entry_df = df.loc[df["entry/exit"] == 1.0]
    exit_df = df.loc[df["entry/exit"] == -1.0]
    entry_marker = marker_trace(entry_df.index, 
                                entry_df.sma10, 
                                'triangle-up', '#0efa0a', 'buy')
    exit_marker = marker_trace(exit_df.index, 
                               exit_df.sma10, 
                               'triangle-down', '#FF0000', 'sell')
    fig.add_traces([line_trace(df, 'sma10', '#fa760a'), 
                    line_trace(df, 'sma20', '#0af7f7'), 
                    entry_marker, 
                    exit_marker])
    fig["layout"] = get_fig_layout()
    return fig


def get_trade_fig(df):
    fig = go.Figure()
    fig.add_traces([line_trace(df, 'entry/exit')])
    fig["layout"] = get_fig_layout()
    return fig

def get_backtest_fig(df, timeframe):
    fig = go.Figure()
    tickformat = {}
    if (timeframe in ['30m','1h','1d','1w']):
        tickformat = {'tickformat':'%Y-%m-%d'}
    entry_df = df.loc[df["Entry/Exit"] == 1.0]
    exit_df = df.loc[df["Entry/Exit"] == -1.0]
    entry_marker = marker_trace(entry_df.index, 
                                entry_df['Portfolio Total'], 
                                'circle', '#15ed24', 'buy', 10)
    exit_marker = marker_trace(exit_df.index, 
                               exit_df['Portfolio Total'], 
                               'circle', '#ed1f3f', 'sell', 8)
    fig.add_traces([line_trace(df, 'Portfolio Total', '#b2c2c0'), 
                    entry_marker, 
                    exit_marker])
    fig["layout"] = get_fig_layout(**tickformat)
    return fig

#app.config.suppress_callback_exceptions = True
@app.callback([Output('crypto-2-symbol', 'data'),
              Output('two-sec-interval', 'disabled'),
              Output('five-sec-interval', 'disabled')],
              [Input('trade-btn', 'n_clicks')],
              [State('crypto-2-select-dropdown', 'value')])
def reinitalize_crypto(n_clicks, crypto):
    if(crypto==None or crypto==''):
        raise PreventUpdate
    crypto_stream.init_connection()
    #data = [{'close':0.0, "balance": 10000, "shares": 0, 'status':''}]
    return crypto, False, False

@app.callback([Output('live-crypto-graph', 'figure'),
              Output('live-signal-graph', 'figure')],
              [Input('two-sec-interval', 'n_intervals')],
              [State('crypto-2-symbol', 'data')])
def update_close_scatter(n, crypto):
    df = crypto_stream.fetch_data(crypto)
    signal_df = crypto_stream.generate_signals(df)
    return get_close_fig(df), get_sma_fig(signal_df)


'''
@app.callback(Output('live-signal-graph', 'figure'),
              [Input('two-sec-interval', 'n_intervals')])
def update_signal_scatter(n):
    time.sleep(1)
    df = crypto_stream.get_data_from_table()
    signal_df = crypto_stream.generate_signals(df)
    return get_sma_fig(signal_df)
'''

@app.callback([Output('stream-table', 'data'),
              Output('entry-exit-dict', 'data'),
              Output('live-trade-graph', 'figure')],
              [Input('five-sec-interval', 'n_intervals')],
              [State('stream-table', 'data'),
              State('entry-exit-dict', 'data')])
def execute_trade(n_intervals, buy_sell_data, entry_exit_df):
    if entry_exit_df:
        entry_exit_df = pd.DataFrame.from_dict(entry_exit_df)
    #entry_exit_df = rf_model.predict(entry_exit_df, 22)
    entry_exit_df = crypto_stream.generate_signals(crypto_stream.get_data_from_table())
    #sprint(len(entry_exit_df))
    if entry_exit_df is None or len(entry_exit_df)<19: 
        raise PreventUpdate
    else:
        account= buy_sell_data[-1]
        account = crypto_stream.execute_trade_strategy(entry_exit_df, account)
        if account==None:
            raise PreventUpdate
        buy_sell_data.append(account)
    return buy_sell_data, entry_exit_df.to_dict('series'), get_trade_fig(entry_exit_df)

'''@app.callback([Output('model-entry-exit', 'data'),
              Output('live-trade-graph', 'figure')],
              [Input('five-sec-interval', 'n_intervals')],
              [State('model-entry-exit', 'data')])
def execute_prediction(n_intervals, entry_exit_df):
    if entry_exit_df:
        entry_exit_df = pd.DataFrame.from_dict(entry_exit_df)
    entry_exit_df = rf_model.predict(entry_exit_df, 22)
    #entry_exit_df = crypto_stream.generate_signals(crypto_stream.get_data_from_table())
    #sprint(len(entry_exit_df))
    if entry_exit_df is None or len(entry_exit_df)<22: 
        raise PreventUpdate
    return entry_exit_df.to_dict('series'), get_trade_fig(entry_exit_df)'''



@app.callback([Output("loading-output-1", "children"),
               Output('backtesting-results-container', 'style'),
               Output('crypto-1-symbol', 'data'),
               Output('trade-metric-table', 'data'),
               Output('backtesting-graph', 'figure'),
               Output('eval_metric_table', 'data')],
               [Input('backtest-btn', 'n_clicks')],
               [State('crypto-1-select-dropdown', 'value'),
               State('model-select-dropdown', 'value'),
               State('timeframe-select-dropdown', 'value'),
               State('initial-capital-input', 'value'),
               State('no-of-shares-input', 'value')])
def reinitalize_model(n_clicks, crypto, model_name, timeframe, initial_capital, no_of_shares):
    portfolio_metrics, trade_metrics, portfolio_evaluation = backtesting.main(crypto, model_name, timeframe, initial_capital, no_of_shares)
    return '', {'display':'block'},crypto, trade_metrics.to_dict("rows"), get_backtest_fig(portfolio_metrics, timeframe), portfolio_evaluation.reset_index().to_dict("rows")


def get_data_table(table_info):
    return dash_table.DataTable(
            id=table_info['id'],
            style_header={"fontWeight": "bold", "color": "inherit"},
            style_as_list_view=True,
            fill_width=True,
            style_cell={
                "backgroundColor": "#1e2130",
                "fontFamily": "Open Sans",
                "padding": "0 2rem",
                "color": "darkgray",
                "border": "none",
            },
            css=[
                {"selector": "tr:hover td", 
                 "rule": "color: #91dfd2 !important;"},
                {"selector": "tr:last-child", 
                 "rule": "display:none !important;"},
                {"selector": "td", 
                 "rule": "border: none !important;"},
                {"selector": ".dash-cell.focused","rule": 
                 "background-color: #1e2130 !important;",
                },
                {"selector": "table", 
                 "rule": "--accent: #1e2130;"},
                {"selector": "tr", 
                 "rule": "background-color: transparent"},
            ],
            data=table_info['data'],
            columns=table_info['columns'])

def get_evaluation_metrics_table(data=[]):
    return dash_table.DataTable(
            id='eval_metric_table',
            style_header={"fontWeight": "bold", "color": "inherit"},
            style_as_list_view=True,
            fill_width=True,
            style_cell_conditional=[
                {"if": {"column_id": "Specs"}, "textAlign": "left"}
            ],
            style_cell={
                "backgroundColor": "#1e2130",
                "fontFamily": "Open Sans",
                "padding": "0 2rem",
                "color": "darkgray",
                "border": "none",
            },
            css=[
                {"selector": "tr:hover td", "rule": "color: #91dfd2 !important;"},
                {"selector": "td", "rule": "border: none !important;"},
                {
                    "selector": ".dash-cell.focused",
                    "rule": "background-color: #1e2130 !important;",
                },
                {"selector": "table", "rule": "--accent: #1e2130;"},
                {"selector": "tr", "rule": "background-color: transparent"},
            ],
            data=data,#new_df.to_dict("rows"),
            columns=[{"id": c, "name": c} for c in ["Metrics", "Backtest"]],
        )

def get_btn_div(id_btn, btn_name):
    return html.Div(
                children=[html.Button(
                    btn_name,
                    id=f"{id_btn}-btn",
                    n_clicks=0
            )])
def get_dropdown(id_name, data_list, value, title):
    return html.Div(
            id=f"{id_name}-select-menu",
            # className='five columns',
            children=[
                html.Label(id=f"{id_name}-select-title", children=f"{title}"),
                dcc.Dropdown(
                    id=f"{id_name}-select-dropdown",
                    options=list(
                        {"label": data, "value": data} for data in data_list
                    ),
                    value=value,
                )])

def get_numeric_input(id_name, value, title):
    return html.Div(
            id=f"{id_name}-menu",
            # className='five columns',
            children=[
                html.Label(id=f"{id_name}-title", children=title),
                daq.NumericInput(
    id=f"{id_name}-input", className="setting-input", value=value, size=200, max=9999999
)])
                
def build_top_panel():
    return html.Div(
        id="top-section-container",
        className="row",
        children=[
            dcc.Store(id='crypto-2-symbol', storage_type='local', data=crypto_stream.SYMBOL),
            dcc.Store(id='entry-exit-dict', storage_type='local'),
            # Metrics summary
            html.Div(
                id="live-data-streaming",
                className="eight columns",
                children=[
                    generate_section_banner("Closing Price"),
                    dcc.Graph(id='live-crypto-graph'),
                    generate_section_banner("Signals"),
                    dcc.Graph(id='live-signal-graph'),
                    generate_section_banner("Trade"),
                    dcc.Graph(id='live-trade-graph', figure={'layout':get_fig_layout()})
                ],
            ),
            # Piechart
            html.Div(
                id="trade-table",
                className="four columns",
                children=[
                    html.Br(),
                    get_dropdown('crypto-2', crypto_stream.get_crypto_symbols(), '', 'Crypto'),
                    html.Br(),
                    get_btn_div('trade', 'Trade'),
                    html.Br(),
                    #get_crypto_dropdown('crypto-2'),
                    generate_section_banner("Trade Data"),
                    get_data_table(STREAM_TABLE)
                ],
            ),
        ],
    )


def build_tab_backtesting():
    return html.Div([
        # Manually select metrics
        html.Div(
            id="set-specs-intro-container",
            # className='twelve columns',
            children=html.P(
                "Use Backtesting, to evaluate the effectiveness of a AI model by running the strategy against historical data "
            )
        ),
        html.Div(
            id="settings-menu",
            children=[
                dcc.Store(id='crypto-1-symbol', storage_type='local', data=crypto_stream.SYMBOL),
                html.Div(
                id="backtesting-settings",
                className="five columns",
                children=[
                    html.Div(
                        className="six columns",
                        children=[
                            html.Br(),
                            get_dropdown('crypto-1', crypto_stream.get_crypto_symbols(), crypto_stream.SYMBOL, 'Crypto'),
                            #get_crypto_dropdown('crypto-1'),
                            html.Br(),
                            get_dropdown('model', backtesting.model_list(), backtesting.model_list()[0], 'Model'),
                            html.Br(),
                            get_numeric_input('no-of-shares', 10, 'No of Shares')
                        ]
                    ),
                    html.Div(
                        className="six columns",
                        children=[
                            html.Br(),
                            get_dropdown('timeframe', ['1m', '5m', '30m', '1h', '1d','1w'], '1m', 'Interval'),
                            html.Br(),
                            get_numeric_input('initial-capital', 100000.0, 'Initial Capital'),
                            html.Br(),
                            html.Br(),
                            get_btn_div('backtest', 'Backtest'),
                            html.Br(),
                            
                        ]
                    )
                ]),
                html.Div(
                    id='loading-div',
                    className="one columns",
                    children=[
                        html.Br(),
                        html.Br(),
                        html.Div(
                                className='ten rows',
                                children=[dcc.Loading(
                                id="loading-1",
                                type="default",
                                children=html.Div(id="loading-output-1")
                                )
                            ]
                        ),
                        html.Br(),
                    ]
                ),
                html.Div(
                    id="backtesting-metrics",
                    className="six columns",
                    children=[
                        generate_section_banner("Portfolio Evaluation Metrics"),
                        html.Br(),
                        get_evaluation_metrics_table()
                    ]
                )
            ]

        ),
        html.Div(
            id="backtesting-results-container",
            style={"display": "none"},
            className='twelve columns',
            children=[
                html.Br(),
                generate_section_banner(" Trading Strategy vs. Backtest Results"),
                dcc.Graph(id='backtesting-graph'),
                html.Br(),
                generate_section_banner("Trade Evaluation Metrics"),
                html.Br(),
                html.Div(id="portfolio-metric-panel", children=[get_data_table(TRADE_METRIC_TABLE),
            ],
        ),
    ])])


def build_tabs():
    return html.Div(
        id="tabs",
        className="tabs",
        children=[
            dcc.Tabs(
                id="app-tabs",
                value="tab1",
                className="custom-tabs",
                children=[
                    dcc.Tab(
                        id="Specs-tab",
                        label="Model Backtesting",
                        value="tab1",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                        children=build_tab_backtesting()
                    ),
                    dcc.Tab(
                        id="Control-chart-tab",
                        label="Control Charts Dashboard",
                        value="tab2",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                        children=build_top_panel()
                    ),
                ],
            )
        ],
    )


def build_banner():
    return html.Div(
        id="banner",
        className="banner",
        children=[
            html.Div(
                id="banner-text",
                children=[
                    html.H5("Mind Bot"),
                    html.H6("An Automated program that buy and sell cryptocurrencies at the right time"),
                ],
            ),
            html.Div(
                id="banner-logo",
                children=[
                    #html.Button(id="learn-more-button", children="LEARN MORE", n_clicks=0),
                    html.Img(id="logo", src=app.get_asset_url("dash-new-logo.png")),
                ],
            ),
        ],
    )




app.layout = html.Div(
    id="big-app-container",
    children=[
        build_banner(),
        # Interval component for live clock
        dcc.Interval(id="two-sec-interval", disabled=True, interval=1 * 1000, n_intervals=0),dcc.Interval(id="five-sec-interval", disabled=True, interval=1 * 1000, n_intervals=0),
        dcc.Interval(
            id="interval-component",
            interval=2 * 1000,  # in milliseconds
            n_intervals=50,  # start at batch 50
            disabled=True,
        ),
        html.Div(
            id="app-container",
            children=[
                build_tabs(),
                # Main app
                html.Div(id="app-content"),
            ],
        )
    ],
)



# Running the server
if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
    