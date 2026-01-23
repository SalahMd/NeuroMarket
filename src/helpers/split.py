import pandas as pd
from src.helpers import config


def time_split(X, y, train_ratio=0.7, val_ratio=0.15):
    n = len(X)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return (
        X[:train_end], y[:train_end],
        X[train_end:val_end], y[train_end:val_end],
        X[val_end:], y[val_end:]
    )


def ticker_split(df, train_ratio=0.6, val_ratio=0.2):
    
    train_dfs = []
    val_dfs = []
    test_dfs = []

    for _, ticker_df in df.groupby("Ticker"):
        
        n = len(ticker_df)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_size = train_end
        val_size = val_end - train_end
        test_size = n - val_end

        if train_size < config.WINDOW_SIZE or \
           val_size < config.WINDOW_SIZE or \
           test_size < 1:
            continue

        train_dfs.append(ticker_df.iloc[:train_end])
        val_dfs.append(ticker_df.iloc[train_end:val_end])
        test_dfs.append(ticker_df.iloc[val_end:])

    if not train_dfs or not val_dfs or not test_dfs:
        raise ValueError("No tickers with sufficient data for all sets.")

    return (
        pd.concat(train_dfs),
        pd.concat(val_dfs),
        pd.concat(test_dfs)
    )
