import pandas as pd

def test(df0, var0):
    df0[var0] = df0[var0].dt.date
    return df0
