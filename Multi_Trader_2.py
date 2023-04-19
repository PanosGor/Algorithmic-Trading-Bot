import pandas as pd
import numpy as np
import tpqoa
from datetime import datetime, timedelta
import time
import pickle
import pymysql
import uuid
from statsmodels.tsa.arima.model import ARIMA

class ConTrader(tpqoa.tpqoa):
    def __init__(self, conf_file, instrument, bar_length, window, units, lags, model,p,ind,q,SMA_S,SMA_L,SMA_Bol,Dev,Comb_Str=1)         :
        super().__init__(conf_file)
        self.instrument = instrument
        self.Comb_Str = Comb_Str
        self.bar_length = pd.to_timedelta(bar_length)
        self.tick_data = pd.DataFrame()
        self.raw_data = None
        self.data = None 
        self.last_bar = None
        self.units = units
        self.position = 0
        self.profits = []

        #*****************add strategy-specific attributes here******************
        self.window = window
        self.p = p
        self.ind = ind
        self.q = q
        self.lags = lags
        self.model = pickle.load(open(model, "rb"))
        self.SMA_S = SMA_S
        self.SMA_L = SMA_L
        self.SMA_Bol = SMA_Bol
        self.Dev = Dev
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
            #self.model_train(1,1,0)
    
    def resample_and_join(self):
        self.raw_data = self.raw_data.append(self.tick_data.resample(self.bar_length, 
                                                                  label="right").last().ffill().iloc[:-1])
        self.tick_data = self.tick_data.iloc[-1:]
        self.last_bar = self.raw_data.index[-1]
        
            
    def define_strategy(self): # "strategy-specific"
        df = self.raw_data.copy()
        
        #******************** Contrarian_Strategy*******************************
        
        df["Contrarian_returns"] = np.log(df[self.instrument] / df[self.instrument].shift())
        df["Contrarian_position"] = -np.sign(df['Contrarian_returns'].rolling(self.window).mean())
        
        #*************************** ML_Strategy *******************************
        
        df = df.append(self.tick_data) # append latest tick (== open price of current bar)
        df["ML_returns"] = np.log(df[self.instrument] / df[self.instrument].shift())
        cols = []
        for lag in range(1, self.lags + 1):
            col = "lag{}".format(lag)
            df[col] = df['ML_returns'].shift(lag)
            cols.append(col)
        df.dropna(inplace = True)
        df["ML_position"] = self.model.predict(df[cols])
        
        #***************************** ARIMA Strategy ***************************
        
        ARIMA_model = ARIMA(df[self.instrument], order=(self.p,self.ind,self.q))
        model_fit = ARIMA_model.fit()
        df['ARIMA_forecast'] = model_fit.predict()
        df['ARIMA_returns'] = df['ARIMA_forecast'] - df[self.instrument].shift(1)
        df.dropna(inplace=True)
        df["ARIMA_position"] = np.sign(df['ARIMA_returns'])
        
        #****************************** SMA Strategy ***************************
        df["SMA_S"] = df[self.instrument].rolling(self.SMA_S).mean()
        df["SMA_L"] = df[self.instrument].rolling(self.SMA_L).mean()
        df.dropna(inplace=True)
        df["SMA_position"] = np.where(df["SMA_S"] > df["SMA_L"], 1.0, -1.0)
        
        #****************************** Bollinger ******************************
        df["SMA_Bol"] = df[self.instrument].rolling(self.SMA_Bol).mean()
        df["Lower_Bol"] = df["SMA_Bol"] - df[self.instrument].rolling(self.SMA_Bol).std() * self.Dev
        df["Upper_Bol"] = df["SMA_Bol"] + df[self.instrument].rolling(self.SMA_Bol).std() * self.Dev
        df.dropna(inplace=True)
        df["distance"] = df[self.instrument] - df['SMA_Bol']
        df["Bol_position"] = np.where(df[self.instrument] < df.Lower_Bol, 1, np.nan)
        df["Bol_position"] = np.where(df[self.instrument] > df.Upper_Bol, -1, df["Bol_position"])
        df["Bol_position"] = np.where(df.distance * df.distance.shift(1) < 0, 0, df["Bol_position"])
        df["Bol_position"] = df.Bol_position.ffill().fillna(0)
        
        #***********************************************************************
        #Unanimous Trade Strategy
        if self.Comb_Str == 1:
            df['position'] = np.where((df['ARIMA_position']==df['ML_position']) & (df['ML_position']==df['Contrarian_returns']) & (df['Contrarian_returns'] == df['SMA_position']),df['ML_position'],0)                          
        elif self.Comb_Str == 2:
            df['position'] = np.sign(df['Bol_position']+ df['ARIMA_position'] + df['ML_position'] + df['Contrarian_returns'] + df['SMA_position'])      
        
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
        self.mycursor.execute(f'CREATE TABLE IF NOT EXISTS TradesTable_{self.instrument} (TradeID CHAR(40) NOT NULL PRIMARY KEY,time DATETIME,Instrument CHAR(40), Units INT(20),Price FLOAT(10),pnl FLOAT(10),CumPnL FLOAT(10))')                                                               
        sql1 = f"INSERT INTO TradesTable_{self.instrument} (TradeID,time,Instrument,Units,Price,pnl,CumPnL) VALUES (%s,%s,%s,%s,%s,%s,%s)"
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
        


# In[2]:

if __name__ == "__main__":
    
    trader = ConTrader("oanda.cfg", "EUR_USD", "1min", window = 1,
                       units = 100000, lags = 5, model = 'logreg.pkl',
                       p = 1, ind = 1,q = 0, SMA_S = 50,SMA_L = 200 ,
                       SMA_Bol=20,Dev=1,Comb_Str=2)                 
    trader.get_most_recent()
    trader.stream_data(trader.instrument, stop = None)
    if trader.position != 0: 
        close_order = trader.create_order(trader.instrument, units = -trader.position * trader.units, 
                                              suppress = True, ret = True) 
        trader.report_trade(close_order, "GOING NEUTRAL")
        trader.position = 0


# In[ ]:




