import numpy as np
import pandas as pd


def make_labels(open_, high, low, close, atr, window, mult, slip=0.0005):
    #Signal generated at the close on day t;
    #enter the market at the open on day t+1.
    #slip :  0.0005 (0.05%)
   
    n = len(close)
    labels = np.full(n, -1, dtype=int)
    o, h, l, a = open_.values, high.values, low.values, atr.values

    for t in range(n - window):                # windowtime: t+1_open to t+window_close
        if np.isnan(a[t]) or a[t] <= 0:
            continue

        entry = o[t + 1]                        # Buy : Open[t+1]
        tau = a[t] * mult + entry * slip         
        # Buy signal  : Open[t+1] + (ATR[t]*mult + Open[t+1]*slip)
        # sell signal  : Open[t+1] - (ATR[t]*mult + Open[t+1]*slip)
        up, dn = entry + tau, entry - tau

        label = 1                               # Hold
        for k in range(1, window + 1):
            hu, hd = h[t + k] >= up, l[t + k] <= dn
            if hu and hd:   label = 2; break    # Sell
            if hu:          label = 0; break
            if hd:          label = 2; break
        labels[t] = label

    return labels

