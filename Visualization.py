import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.finance import candlestick_ohlc

def plotSMA3(df):
    df['SMA3'].plot()
    plt.title('Three days moving average')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

def ohlc(df):
    df_ohlc = df[['Open','High','Low','Close']]
    df_ohlc.reset_index(inplace = True)
    df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)


    ax1 = plt.subplot2grid((6,1),(0,0), rowspan = 6, colspan = 1)

    ax1.set_title('Ohlc information for stock')

    ax1.xaxis_date()
    candlestick_ohlc(ax1, df_ohlc.values, width = 0.5, colorup = 'r', colordown = 'g')

    plt.xlabel('Date')
    plt.ylabel('Price')

    plt.show() 
        

def ohlcVolume(df):  
    df_ohlc = df[['Open','High','Low','Close']]
    df_volume = df['Volume']
    df_ohlc.reset_index(inplace = True)
    df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)

    ax1 = plt.subplot2grid((6,1),(0,0), rowspan = 5, colspan = 1)
    ax2 = plt.subplot2grid((6,1),(5,0), rowspan = 1, colspan = 1, sharex = ax1)

    ax1.xaxis_date()
    candlestick_ohlc(ax1, df_ohlc.values, width = 0.5, colorup = 'r', colordown = 'g')

    ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)

    plt.show()

def closeLabel(df, ticker):
    df_Close = df['Adj Close']
    df_Prediction = df['Prediction']
    df_Actual = df['Actual Trend']

    ax1 = plt.subplot2grid((6,1),(0,0), rowspan = 5, colspan = 1)
    ax1.set_title(ticker)
    ax2 = plt.subplot2grid((6,1),(5,0), rowspan = 1, colspan = 1, sharex = ax1)

    ax1.plot(df_Close)
    ax2.plot(df_Prediction, label = 'Prediction Trend')
    ax2.plot(df_Actual, label = 'Actual Trend')
    
    plt.legend()
    plt.show()
