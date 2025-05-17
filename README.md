# NLP Sentiment Analysis for Stock Trends

This project aims to perform sentiment analysis on Twitter data related to stocks and use the sentiment to simulate a simple trading strategy. The project will compare three different NLP models for sentiment classification:

1.  Logistic Regression
2.  BERT (Bidirectional Encoder Representations from Transformers)
3.  GPT-2 (Generative Pre-trained Transformer 2)

## Datasets

-   **Unlabeled Tweets**: 900k+ tweets from [StephanAkkerman/stock-market-tweets-data](https://huggingface.co/datasets/StephanAkkerman/stock-market-tweets-data)
-   **Labeled Tweets**: ~30k tweets with sentiment labels (positive, negative, neutral) from [TimKoornstra/financial-tweets-sentiment](https://huggingface.co/datasets/TimKoornstra/financial-tweets-sentiment)

## Methodology

1.  **Data Loading and Preprocessing**: Tweets will be loaded, cleaned, and tokenized.
2.  **Model Training**: The three models will be trained/fine-tuned on the labeled dataset.
3.  **Sentiment Prediction**: Trained models will predict sentiment on new or historical tweets.
4.  **Trading Strategy Simulation**: A simple strategy (buy on positive, sell on negative) will be simulated using stock data fetched via `yfinance`.
5.  **Visualization**: Results, including stock price charts with buy/sell signals, will be plotted.

## Project Goals

-   Implement functional code for each stage.
-   Ensure the logic and approach are sound.
-   Produce clear and informative visualizations. 