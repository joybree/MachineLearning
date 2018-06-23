# -*- coding: utf-8 -*-
"""
Created on Sun May 27 21:02:34 2018

@author: TH
"""

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
import numpy as np
from datetime import datetime
import pylab as pl

###Global varible

dir = "E:/code/python/MachineLearning/data/"
pre_num = 1

def prediction(ts_data, pre_num=1):
    para = (0,0,1)
    model = ARIMA(ts_data[0:len(ts_data)-pre_num], para).fit()
    prediction_value, prediction_var, prediction_con= model.forecast(pre_num)
    return (prediction_value, prediction_var, prediction_con)

if __name__ == "__main__":
    df_data = pd.read_csv(dir+'ts_data.csv', encoding='utf-8', index_col='date')
    df_data.index = pd.to_datetime(df_data.index)
    ts_data = df_data['value']
    prediction_value, prediction_var, prediction_con = prediction(ts_data, pre_num)
    print(prediction_value)
    
    