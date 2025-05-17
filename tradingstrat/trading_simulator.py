import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from data_loader import load_unlabeled_tweets
from stock_data_handler import get_stock_data, extract_cashtags
from predictor import predict_logistic_regression, predict_bert, predict_gpt2
from text_preprocessor import clean_tweet_text

pd.options.mode.chained_assignment = None # default='warn' 

TARGET_STOCKS_FOR_MARKET_SENTIMENT = [
    "MSFT", "AAPL", "AMZN", "META", "GOOGL", "GOOG", "JNJ", "JPM", "V", "PG", "MA", 
    "INTC", "UNH", "BAC", "T", "HD", "XOM", "DIS", "VZ", "KO", "MRK", 
    "CMCSA", "CVX", "PEP", "PFE", "BRK-B", "TSLA", "NVDA"
]

MARKET_INDEX_TICKER = "^GSPC" # S&P 500

BUY_THRESHOLD = 0.2  # If avg_daily_sentiment >= this, Buy
SELL_THRESHOLD = 0 # If avg_daily_sentiment <= this, Sell

def plot_average_daily_sentiment(daily_sentiment_df, model_type, start_date, end_date):
    plt.figure(figsize=(15, 5))
    plt.plot(daily_sentiment_df['date'], daily_sentiment_df['avg_daily_sentiment'], label='Average Market Sentiment', marker='.', linestyle='-')
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.axhline(BUY_THRESHOLD, color='lightgreen', linestyle=':', linewidth=0.8, label=f'Buy Threshold ({BUY_THRESHOLD})')
    plt.axhline(SELL_THRESHOLD, color='lightcoral', linestyle=':', linewidth=0.8, label=f'Sell Threshold ({SELL_THRESHOLD})')
    plt.ylim(-1.05, 1.05)
    plt.title(f'Average Daily Market Sentiment ({model_type.capitalize()} Model)')
    plt.xlabel('Date')
    plt.ylabel('Average Sentiment Score (-1 to 1)')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_filename = f'results/{model_type}/{MARKET_INDEX_TICKER.replace("^", "")}_avg_daily_sentiment_{model_type}_{start_date}_to_{end_date}.png'
    plt.savefig(plot_filename)
    print(f"Average daily sentiment plot saved as {plot_filename}")
    plt.close()

def run_market_simulation(start_date, end_date, model_type, initial_capital=100.0, tweet_sample_fraction=0.05):
    """
    Runs the market trading simulation based on aggregated sentiment from key stocks.
    Args:
        model_type (str): 'logistic', 'bert', or 'gpt2'.
        tweet_sample_fraction (float): Fraction of relevant tweets to sample for simulation.
    """
    print(f"Starting S&P 500 simulation from {start_date} to {end_date} using {model_type} model.")
    print(f"Using {tweet_sample_fraction*100:.2f}% of relevant tweets for simulation.")

    # 1. Load and filter tweets for the relevant period
    print("Loading tweets...")
    unlabeled_df = load_unlabeled_tweets()
    unlabeled_df['created_at'] = pd.to_datetime(unlabeled_df['created_at']).dt.tz_localize(None)
    
    tweets_df = unlabeled_df[(
        unlabeled_df['created_at'] >= pd.to_datetime(start_date)) & 
        (unlabeled_df['created_at'] <= pd.to_datetime(end_date) + pd.Timedelta(days=1))
    ]

    tweets_df['cleaned_text'] = tweets_df['text'].apply(clean_tweet_text)
    tweets_df['cashtags'] = tweets_df['text'].apply(lambda x: [tag.upper() for tag in extract_cashtags(x)])
    
    # Filter tweets that mention any of the target stocks
    def contains_target_stock(extracted_cashtags):
        if not extracted_cashtags: return False
        return any(tag in TARGET_STOCKS_FOR_MARKET_SENTIMENT for tag in extracted_cashtags)

    relevant_tweets = tweets_df[tweets_df['cashtags'].apply(contains_target_stock)].copy()

    if relevant_tweets.empty:
        print(f"No tweets found mentioning target stocks ({TARGET_STOCKS_FOR_MARKET_SENTIMENT}) in the specified date range.")
        return

    print(f"Found {len(relevant_tweets)} relevant tweets for market sentiment analysis.")

    if not relevant_tweets.empty and tweet_sample_fraction < 1.0:
        relevant_tweets = relevant_tweets.sample(frac=tweet_sample_fraction, random_state=42) # Add random_state for reproducibility
        print(f"Sampled down to {len(relevant_tweets)} tweets ({tweet_sample_fraction*100:.2f}% of original relevant).")

    # 2. Predict sentiment for these tweets
    print("Predicting sentiment for relevant tweets...")
    tweet_texts = relevant_tweets['cleaned_text'].tolist()
    if not tweet_texts:
        print("No text available in relevant tweets for sentiment prediction.")
        return

    sentiments_map = {
        'logistic': predict_logistic_regression,
        'bert': predict_bert,
        'gpt2': predict_gpt2
    }
    # Initial predictions: 0 (Neutral), 1 (Positive), 2 (Negative)
    raw_sentiments = sentiments_map[model_type](tweet_texts)
    relevant_tweets.loc[:, 'raw_sentiment'] = raw_sentiments
    relevant_tweets = relevant_tweets[relevant_tweets['raw_sentiment'].notna()]
    relevant_tweets.loc[:, 'raw_sentiment'] = relevant_tweets['raw_sentiment'].astype(int)

    # Remap sentiments for averaging: Positive (1) -> 1, Neutral (0) -> 0, Negative (2) -> -1
    sentiment_remapping = {0: 0, 1: 1, 2: -1}
    relevant_tweets['sentiment_for_avg'] = relevant_tweets['raw_sentiment'].map(sentiment_remapping)
    # relevant_tweets['sentiment_for_avg'] = (relevant_tweets['sentiment_for_avg'] - .8 * relevant_tweets['sentiment_for_avg'].mean()) * 1.6

    daily_avg_sentiment_df = relevant_tweets.groupby(relevant_tweets['created_at'].dt.date)['sentiment_for_avg'].mean().reset_index()
    daily_avg_sentiment_df.rename(columns={'created_at': 'date', 'sentiment_for_avg': 'avg_daily_sentiment'}, inplace=True)
    daily_avg_sentiment_df['avg_daily_sentiment'] = (daily_avg_sentiment_df['avg_daily_sentiment'] - 0.15) * 1.6
    daily_avg_sentiment_df['date'] = pd.to_datetime(daily_avg_sentiment_df['date'])
    print("Average daily market sentiments calculated.")
    plot_average_daily_sentiment(daily_avg_sentiment_df, model_type, start_date, end_date)

    # 4. Fetch S&P 500 data
    print(f"Fetching market index data ({MARKET_INDEX_TICKER})...")
    market_data = get_stock_data(MARKET_INDEX_TICKER, start_date, pd.to_datetime(end_date) + pd.Timedelta(days=1))
    if market_data.empty:
        print(f"Could not fetch market index data for {MARKET_INDEX_TICKER}.")
        return
    market_data.index = market_data.index.tz_localize(None)

    # 5. Combine S&P 500 data with market signals
    trading_df = market_data.merge(daily_avg_sentiment_df, left_index=True, right_on='date', how='left')
    trading_df.set_index('date', inplace=True, drop=False)
    trading_df['avg_daily_sentiment'] = trading_df['avg_daily_sentiment'].fillna(0) # Treat no-tweet days as neutral average

    # Signal: 1 for Buy, 2 for Sell, 0 for Neutral
    def sentiment_to_strategy_signal(avg_sentiment):
        if avg_sentiment >= BUY_THRESHOLD: return 1
        if avg_sentiment <= SELL_THRESHOLD: return -1
        return 0
    trading_df['strategy_signal'] = trading_df['avg_daily_sentiment'].apply(sentiment_to_strategy_signal)

    # Plot 1: S&P 500 with Derived Market Signals
    print("Plotting S&P 500 with market signals...")
    plt.figure(figsize=(15, 7))
    plt.plot(trading_df.index, trading_df['Close'], label=f'{MARKET_INDEX_TICKER} Close Price', alpha=0.7)

    buy_signals = trading_df[trading_df['strategy_signal'] == 1]
    plt.scatter(buy_signals.index, buy_signals['Close'], label=f'Buy Signal (Avg Sent >= {BUY_THRESHOLD})', marker='^', color='green', s=100, alpha=1)

    sell_signals = trading_df[trading_df['strategy_signal'] == -1]
    plt.scatter(sell_signals.index, sell_signals['Close'], label=f'Sell Signal (Avg Sent <= {SELL_THRESHOLD})', marker='v', color='red', s=100, alpha=1)

    plt.title(f'{MARKET_INDEX_TICKER} with Derived Sentiment Signals ({model_type.capitalize()} Model)')
    plt.xlabel('Date')
    plt.ylabel('Index Close Price')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plot_filename = f'results/{model_type}/{MARKET_INDEX_TICKER.replace("^", "")}_price_derived_signals_{model_type}_{start_date}_to_{end_date}.png'
    plt.savefig(plot_filename)
    print(f"Price with derived signals plot saved as {plot_filename}")

    # Trading Strategy Simulation
    trading_df['position'] = 0 # 1 for long, 0 for no position
    trading_df['strategy_cash'] = initial_capital
    trading_df['strategy_holdings_value'] = 0.0
    trading_df['strategy_portfolio_value'] = initial_capital
    shares_held = 0
    in_market = False

    for i in range(len(trading_df)):
        current_signal = trading_df['strategy_signal'].iloc[i]
        current_price = trading_df['Close'].iloc[i]
        current_cash = trading_df['strategy_cash'].iloc[i-1] if i > 0 else initial_capital
        
        if not in_market and current_signal == 1: # Buy Signal
            shares_to_buy = current_cash / current_price
            shares_held = shares_to_buy
            trading_df.loc[trading_df.index[i], 'strategy_cash'] = 0
            trading_df.loc[trading_df.index[i], 'position'] = 1
            in_market = True
        elif in_market and (current_signal == 2 or current_signal == 0): # Sell or Neutral Signal
            trading_df.loc[trading_df.index[i], 'strategy_cash'] = shares_held * current_price
            shares_held = 0
            trading_df.loc[trading_df.index[i], 'position'] = 0
            in_market = False
        else: # Hold or no action
            trading_df.loc[trading_df.index[i], 'strategy_cash'] = current_cash
            if in_market:
                 trading_df.loc[trading_df.index[i], 'position'] = 1

        trading_df.loc[trading_df.index[i], 'strategy_holdings_value'] = shares_held * current_price
        trading_df.loc[trading_df.index[i], 'strategy_portfolio_value'] = trading_df['strategy_cash'].iloc[i] + trading_df['strategy_holdings_value'].iloc[i]

    # Buy and Hold Strategy
    trading_df['buy_hold_portfolio_value'] = initial_capital * (trading_df['Close'] / trading_df['Close'].iloc[0])

    # Plot 2: Portfolio Value Comparison
    plt.figure(figsize=(15, 7))
    plt.plot(trading_df.index, trading_df['strategy_portfolio_value'], label=f'Sentiment Strategy ({model_type.capitalize()})')
    plt.plot(trading_df.index, trading_df['buy_hold_portfolio_value'], label='Buy and Hold')
    plt.title(f'Trading Strategy Performance vs. Buy and Hold ({MARKET_INDEX_TICKER})')
    plt.xlabel('Date'); plt.ylabel('Portfolio Value'); plt.legend(); plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator()); plt.xticks(rotation=45); plt.tight_layout()
    plot_filename = f'results/{model_type}/{MARKET_INDEX_TICKER.replace("^", "")}_portfolio_performance_{model_type}_{start_date}_to_{end_date}.png'
    plt.savefig(plot_filename); print(f"Portfolio performance plot saved as {plot_filename}"); plt.close()

if __name__ == '__main__':
    SIM_START_DATE = "2020-06-01"
    SIM_END_DATE = "2020-07-15" # Extended end date for more data points
    
    # print("\n--- Running Market Simulation with Logistic Regression ---")
    # run_market_simulation(SIM_START_DATE, SIM_END_DATE, model_type='logistic', tweet_sample_fraction=0.95)
    # print("\n--- Running Market Simulation with BERT ---")
    # run_market_simulation(SIM_START_DATE, SIM_END_DATE, model_type='bert', tweet_sample_fraction=0.01)
    print("\n--- Running Market Simulation with GPT-2 ---")
    run_market_simulation(SIM_START_DATE, SIM_END_DATE, model_type='gpt2', tweet_sample_fraction=0.025)
    print("\nAll market simulations complete.") 