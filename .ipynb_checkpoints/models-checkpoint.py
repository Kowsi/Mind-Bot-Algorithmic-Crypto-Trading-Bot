import pandas as pd
import numpy as np
import crypto_stream
import rf_model
import rf_model_2
import warnings
warnings.filterwarnings('ignore')

MODEL_LIST = ['SMA10', 'Random Forest Classifier - 1', 'Random Forest Classifier - 2']

def model_list():
    return MODEL_LIST

def get_models(model_name):
    if(model_name=='Random Forest Classifier - 1'):
        return rf_model
    if(model_name=='Random Forest Classifier - 2'):
        return rf_model_2
    return rf_model


def predict(df_ee, model_name, no_of_data=22):
    model = get_models(model_name)
    #model = package.load_model()
    past_df = crypto_stream.get_data_from_table(no_of_data)
    print(len(past_df))
    past_df = model.get_trading_singals(past_df)
    data = past_df.tail(2)[model.get_statergies()]
    predictions = model.load_model().predict(data)
    entry_exit = predictions[1]-predictions[0]
    df_ee.loc[df_ee.shape[0]-1:,['entry/exit']]=entry_exit
    if(entry_exit!=0):
        print(f'-----------------df_ee---{entry_exit}')
        print(df_ee)
    return df_ee
    