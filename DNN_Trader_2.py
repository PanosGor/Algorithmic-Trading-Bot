import pandas as pd
import numpy as np
import tpqoa
from datetime import datetime, timedelta
import time
import pickle
import pymysql
import uuid
from statsmodels.tsa.arima.model import ARIMA
from tensorflow import keras

class DNNTrader2(tpqoa.tpqoa):
    def __init__(self, conf_file, instrument, bar_length, window, lags, LR_model, DNN_model , RF_model , mu, std, units , Comb_Str):
        super().__init__(conf_file)
        self.instrument = instrument
        self.bar_length = pd.to_timedelta(bar_length)
        self.tick_data = pd.DataFrame()
        self.raw_data = None
        self.data = None 
        self.last_bar = None
        self.units = units
        self.position = 0
        self.profits = []
        self.Comb_Str = Comb_Str
        #*****************add strategy-specific attributes here******************
        self.window = window
        self.lags = lags
        self.LR_model = LR_model
        self.DNN_model = DNN_model 
        self.RF_model = RF_model
        self.mu = mu
        self.std = std
        #************************************************************************
    
    def get_most_recent(self, days = 5):
        while True:
            time.sleep(2)
            now = datetime.utcnow()
            now = now - timedelta(microseconds = now.microsecond)
            past = now - timedelta(days = days)
            df = self.get_history(instrument = self.instrument, start = past, end = now,
                                   granularity = "S5", price = "M", localize = False).c.dropna().to_frame()
            df.rename(columns = {"c":self.instrument}, inplace = True)
            df = df.resample(self.bar_length, label = "right").last().dropna().iloc[:-1]
            self.raw_data = df.copy()
            self.last_bar = self.raw_data.index[-1]
            if pd.to_datetime(datetime.utcnow()).tz_localize("UTC") - self.last_bar < self.bar_length:
                self.start_time = pd.to_datetime(datetime.utcnow()).tz_localize("UTC") # NEW -> Start Time of Trading Session
                break
                
    def on_success(self, time, bid, ask):
        print(self.ticks, end = " ", flush = True)
        
        recent_tick = pd.to_datetime(time)
        df = pd.DataFrame({self.instrument:(ask + bid)/2}, 
                          index = [recent_tick])
        self.tick_data = self.tick_data.append(df)
        
        if recent_tick - self.last_bar > self.bar_length:
            self.resample_and_join()
            self.define_strategy()
            self.execute_trades()
    
    def resample_and_join(self):
        self.raw_data = self.raw_data.append(self.tick_data.resample(self.bar_length, 
                                                                  label="right").last().ffill().iloc[:-1])
        self.tick_data = self.tick_data.iloc[-1:]
        self.last_bar = self.raw_data.index[-1]
    
    def define_strategy(self): # "strategy-specific"
        df = self.raw_data.copy()
        
        #******************** define your strategy here ************************
        #create features
        df = df.append(self.tick_data) # append latest tick (== open price of current bar)
        df["returns"] = np.log(df[self.instrument] / df[self.instrument].shift())
        df["dir"] = np.where(df["returns"] > 0, 1, -1)
        df["sma"] = df[self.instrument].rolling(self.window).mean() - df[self.instrument].rolling(150).mean()
        df["boll"] = (df[self.instrument] - df[self.instrument].rolling(self.window).mean()) / df[self.instrument].rolling(self.window).std()
        df["min"] = df[self.instrument].rolling(self.window).min() / df[self.instrument] - 1
        df["max"] = df[self.instrument].rolling(self.window).max() / df[self.instrument] - 1
        df["mom"] = df["returns"].rolling(3).mean()
        df["vol"] = df["returns"].rolling(self.window).std()
        df.dropna(inplace = True)
        
        # create lags
        self.cols = []
        features = ["dir", "sma", "boll", "min", "max", "mom", "vol"]

        for f in features:
            for lag in range(1, self.lags + 1):
                col = "{}_lag_{}".format(f, lag)
                df[col] = df[f].shift(lag)
                self.cols.append(col)
        df.dropna(inplace = True)
        
        # standardization
        df_s = (df - self.mu) / self.std
        # predict
        df["proba"] = self.DNN_model.predict(df_s[self.cols])
        
        #determine positions
        #************************ logistic Regression Model ********************
        df["LR_position"] = self.LR_model.predict(df_s[self.cols])
        
        #************************ Random Forest Model **************************
        df["RF_position"] = self.RF_model.predict(df_s[self.cols])
        
        #**************************** DNN _ Method *****************************
        df = df.loc[self.start_time:].copy() # starting with first live_stream bar (removing historical bars)
        df["DNN_position"] = np.where(df.proba < 0.50, -1, np.nan)
        df["DNN_position"] = np.where(df.proba > 0.51, 1, df.DNN_position)
        df["DNN_position"] = df.DNN_position.ffill().fillna(0) # start with neutral position if no strong signal
        
        #************************************************************************  
        if self.Comb_Str == 1:
            df['position'] = np.where((df['LR_position']==df['DNN_position']) & (df['DNN_position'] == df['RF_position']),df['DNN_position'],0)                                                    
        elif self.Comb_Str == 2:
            df['position'] = np.sign(df['LR_position'] + df['DNN_position'] + df['RF_position'])  
        
        self.data = df.copy()
    
    def execute_trades(self):
        if self.data["position"].iloc[-1] == 1:
            if self.position == 0:
                order = self.create_order(self.instrument, self.units, suppress = True, ret = True)
                self.report_trade(order, "GOING LONG")
            elif self.position == -1:
                order = self.create_order(self.instrument, self.units * 2, suppress = True, ret = True) 
                self.report_trade(order, "GOING LONG")
            self.position = 1
        elif self.data["position"].iloc[-1] == -1: 
            if self.position == 0:
                order = self.create_order(self.instrument, -self.units, suppress = True, ret = True)
                self.report_trade(order, "GOING SHORT")
            elif self.position == 1:
                order = self.create_order(self.instrument, -self.units * 2, suppress = True, ret = True)
                self.report_trade(order, "GOING SHORT")
            self.position = -1
        elif self.data["position"].iloc[-1] == 0: 
            if self.position == -1:
                order = self.create_order(self.instrument, self.units, suppress = True, ret = True) 
                self.report_trade(order, "GOING NEUTRAL")
            elif self.position == 1:
                order = self.create_order(self.instrument, -self.units, suppress = True, ret = True)
                self.report_trade(order, "GOING NEUTRAL")
            self.position = 0
            
    def date_convert(self,date_str):
        x = date_str.replace('T','/').split('/')
        year = x[0].split('-')[0]
        month = x[0].split('-')[1]
        day = x[0].split('-')[2]
        hour = x[1].split(':')[0]
        minute = x[1].split(':')[1]
        seconds = x[1].split(':')[2][:2]
        full_date = year + '/' + month + '/' + day + ' ' + hour + ':' + minute + ':' + seconds
        new_date = datetime.strptime(full_date,'%Y/%m/%d %H:%M:%S')
        return new_date
    
    def SQL_DB(self,time, units, price, pl, cumpl):
        trade_id = uuid.uuid4()
        time_fin = self.date_convert(str(time))
        self.mydb = pymysql.connect(host="localhost",user="root",passwd="password")
        self.mycursor = self.mydb.cursor()
        self.mycursor.execute("CREATE DATABASE IF NOT EXISTS TradesDB")
        self.mycursor.execute('USE TradesDB')
        self.mycursor.execute(f'CREATE TABLE IF NOT EXISTS TradesTable_DNN_{self.instrument} (TradeID CHAR(40) NOT NULL PRIMARY KEY,time DATETIME,Instrument CHAR(40), Units INT(20),Price FLOAT(10),pnl FLOAT(10),CumPnL FLOAT(10))')                                                               
        sql1 = f"INSERT INTO TradesTable_DNN_{self.instrument} (TradeID,time,Instrument,Units,Price,pnl,CumPnL) VALUES (%s,%s,%s,%s,%s,%s,%s)"
        trade = (trade_id, time_fin, self.instrument, units, price, pl, cumpl)
        self.mycursor.execute(sql1,trade)
        self.mydb.commit()
    
    def report_trade(self, order, going):
        time = order["time"]
        units = order["units"]
        price = order["price"]
        pl = float(order["pl"])
        self.profits.append(pl)
        cumpl = sum(self.profits)
        print("\n" + 100* "-")
        print("{} | {}".format(time, going))
        print("{} | units = {} | price = {} | P&L = {} | Cum P&L = {}".format(time, units, price, pl, cumpl))
        print(100 * "-" + "\n")
        self.SQL_DB(time, units, price, pl, cumpl)


# In[4]:

if __name__ == "__main__":

    DNN = keras.models.load_model('DNN_model_3')
    LR = pickle.load(open('Logistic_Regression_model2.sav', "rb"))
    RF = pickle.load(open('Random_Forest_3.sav', "rb"))
    params = pickle.load(open("params.pkl", "rb"))
    mu = params['mu']
    std = params['std']
    
    
    # In[16]:
    
    
    trader = DNNTrader2("oanda.cfg", "EUR_USD", bar_length = "1min",
                       window = 50, lags = 5, LR_model = LR, DNN_model = DNN, RF_model = RF, mu = mu, std = std, 
                        units = 100000, Comb_Str = 2)
    
    trader.get_most_recent()
    trader.stream_data(trader.instrument, stop = 1000)
    if trader.position != 0:
        close_order = trader.create_order(trader.instrument, units = -trader.position * trader.units,
                                          suppress = True, ret = True) 
        trader.report_trade(close_order, "GOING NEUTRAL")
        trader.position = 0


# In[ ]:




