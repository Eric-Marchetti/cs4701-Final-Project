import yfinance as yf
import pandas as pd
import re

def get_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    # yfinance end_date is exclusive, so add one day to include the end_date
    hist = stock.history(start=start_date, end=end_date)
    if hist.empty:
        print(f"No data found for {ticker} between {start_date} and {end_date}")
        return pd.DataFrame()
    return hist

def extract_cashtags(text):
    """Extracts cashtags (e.g., $AAPL, $TSLA) from a text."""
    if not isinstance(text, str):
        return []
    # Regex to find words starting with $ followed by 1 to 6 uppercase letters (common for tickers)
    # or common indices like $SPX, $NDX. Handles tickers like $BRK.B
    # Ensures cashtag is not followed by another alphanumeric character.
    cashtags = re.findall(r'\$([A-Z]{1,6}(?:\.[A-Z])?|SPX|NDX|DJIA|VIX)(?![A-Za-z0-9])', text.upper()) 
    return list(set(cashtags)) # Return unique uppercase cashtags