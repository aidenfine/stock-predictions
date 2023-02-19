from pandas_datareader import data as pdr
import yfinance as yf

from datetime import datetime

yf.pdr_override()
company = "AAPL"

start = datetime(2020,1,1)
end = datetime(2023,1,1)
data = pdr.get_data_yahoo(company, start=start, end=end)


data.to_csv(f"{company}_stock_data.csv")