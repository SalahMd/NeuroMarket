import pandas as pd
from src.helpers import config


def ticker_split(df, train_ratio=0.6, val_ratio=0.2):
    train, val, test = [], [], []

    for _, g in df.groupby("Ticker"):
        n = len(g)
        tr = int(n * train_ratio)
        vr = int(n * (train_ratio + val_ratio))

        if tr < config.WINDOW_SIZE or vr - tr < config.WINDOW_SIZE:
            continue

        train.append(g.iloc[:tr])
        val.append(g.iloc[tr:vr])
        test.append(g.iloc[vr:])

    return (
        pd.concat(train),
        pd.concat(val),
        pd.concat(test)
    )
