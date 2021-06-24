from typing import final
from flask import Flask,jsonify
from flask import session
from flask import render_template
from flask.globals import request
from pandas import DataFrame as df
import pandas as pd                     # 引用套件並縮寫為 pd
import numpy as np
import pymysql.cursors
import sqlite3
import json
import time
import datetime
import requests
import yfinance as yf
import gzip
import pickle
from pandas_datareader import data
import tensorflow as tf
import sklearn
from scipy.stats import linregress
#import tensorflow as tf

app = Flask(__name__)
stocklist=['2330.TW','2454.TW','2303.TW','aapl','goog']

def predict(num):
    from tensorflow import keras
    model = tf.keras.models.load_model('./model/myModel.h5')
    tickers = num
    start_date = '2002-06-14'
    end_date = '2020-12-30'
    stock_data = data.get_data_yahoo(tickers, start=start_date, end=end_date)
    close_prices = stock_data.iloc[:, 1:2].values
    all_bussinessdays = pd.date_range(start=start_date, end=end_date, freq='B')
    #print(all_bussinessdays)
    close_prices = stock_data.reindex(all_bussinessdays)
    close_prices = stock_data.fillna(method='ffill')
    training_set = close_prices.iloc[:, 1:2].values
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    testing_start_date = '2020-12-31'
    testing_end_date = '2021-06-14'
    test_stock_data = data.get_data_yahoo(tickers, start=testing_start_date, end=testing_end_date)
    test_stock_data_processed = test_stock_data.iloc[:, 1:2].values
    print(test_stock_data_processed.shape)
    all_stock_data = pd.concat((stock_data['Close'], test_stock_data['Close']), axis = 0)
    inputs = all_stock_data[len(all_stock_data) - len(test_stock_data) - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(60, 164):
        X_test.append(inputs[i-60:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #print(X_test)
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    import codecs, json 
    finalprice=predicted_stock_price.tolist()
    refinalprice=test_stock_data_processed.tolist()

    modelten = tf.keras.models.load_model('./model/myModel10.h5')
    predicted_stock_price_ten = modelten.predict(X_test)
    predicted_stock_price_ten = sc.inverse_transform(predicted_stock_price_ten)
    Y=predicted_stock_price_ten[103]
    X=X=[1,2,3,4,5,6,7,8,9,10]
    reg_up = linregress(x = X,y = Y)
    print(reg_up.slope)
    slope=reg_up.slope
    #print(finalprice)
    final=[]
    final.append(num+" 模型效能")
    final.append(finalprice)
    final.append(refinalprice)
    final.append(slope)
    
    return final
@app.route("/get2454")
def get2454():
    conn = sqlite3.connect('nData.db')
    c = conn.cursor()
    cursor = c.execute("SELECT Date,Open,High,Low,Close,Volume from '2454.TW'")
    
    list = []  
    list.append(['聯發科',2454])
    for row in cursor:
       s=datetime.datetime.strptime(str(row[0]), "%Y-%m-%d %H:%M:%S")  
       list.append([s,row[1],row[2],row[3],row[4],row[5]])
      

    return jsonify(list)

@app.route("/get2303")
def get2303():
    conn = sqlite3.connect('nData.db')
    c = conn.cursor()
    cursor = c.execute("SELECT Date,Open,High,Low,Close,Volume from '2303.TW'")
    
    list = []  
    list.append(['聯電',2303])
    for row in cursor:
       s=datetime.datetime.strptime(str(row[0]), "%Y-%m-%d %H:%M:%S")  
       list.append([s,row[1],row[2],row[3],row[4],row[5]])
      

    return jsonify(list)


@app.route("/get2330",methods=['GET','POST'])
def getData():
    conn = sqlite3.connect('nData.db')
    c = conn.cursor()
    cursor = c.execute("SELECT Date,Open,High,Low,Close,Volume from '2330.TW'")
   
    list = []  
    list.append(['台積電',2330])
    for row in cursor:
       s=datetime.datetime.strptime(str(row[0]), "%Y-%m-%d %H:%M:%S")
       list.append([s,row[1],row[2],row[3],row[4],row[5]])
		
    return jsonify(list)

@app.route("/getApple",methods=['GET','POST'])
def getApple():
    conn = sqlite3.connect('nData.db')
    c = conn.cursor()
    cursor = c.execute("SELECT Date,Open,High,Low,Close,Volume from 'aapl'")
   
    list = []  
    list.append(['Apple','Apple'])
    for row in cursor:
       s=datetime.datetime.strptime(str(row[0]), "%Y-%m-%d %H:%M:%S")
       list.append([s,row[1],row[2],row[3],row[4],row[5]])
		
    return jsonify(list)

@app.route("/getGoogle",methods=['GET','POST'])
def getGoogle():
    conn = sqlite3.connect('nData.db')
    c = conn.cursor()
    cursor = c.execute("SELECT Date,Open,High,Low,Close,Volume from 'goog'")
   
    list = []  
    list.append(['Google','google'])
    for row in cursor:
       s=datetime.datetime.strptime(str(row[0]), "%Y-%m-%d %H:%M:%S")
       list.append([s,row[1],row[2],row[3],row[4],row[5]])
		
    return jsonify(list)

@app.route("/")
def test():
	return render_template("index.html")

@app.route("/predict2330")
def predict2330():
    result=predict('2330.TW')
    return jsonify(result)

@app.route("/predict2454")
def predict2454():
    result=predict('2454.TW')
    return jsonify(result)

@app.route("/predict2303")
def predict2030():
    result=predict('2303.TW')
    return jsonify(result)

@app.route("/predictApple")
def predictApple():
    result=predict('aapl')
    return jsonify(result)

@app.route("/predictGoogle")
def predictGoogle():
    result=predict('goog')
    return jsonify(result)
@app.route("/stockdemo")
def stock():
    for i in range(0,5):
        stockid=stocklist[i]
        data=yf.Ticker(stockid)
        df=data.history(period="2y")
        con=sqlite3.connect("nData.db")
        df.to_sql(stockid,con,if_exists='replace')

    return render_template("stockdemo.html")


if __name__ == "__main__":
	port = 8000
	app.run(host='0.0.0.0', port=port)