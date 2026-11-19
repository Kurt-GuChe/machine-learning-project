import numpy as np
import torch
from torch import nn


class LSTMAttention(nn.Module):

    def __init__(
        self,
        n_feat=15,
        n_class=3,
        hid=64,
        heads=4,
        drop=0.3,
    ):
        super().__init__()

        # Bidirectional LSTM: n_feat -> 2 * hid
        self.lstm1 = nn.LSTM(
            input_size=n_feat,
            hidden_size=hid,
            batch_first=True,
            bidirectional=True,
        )

        # Unidirectional LSTM: 2 * hid -> 2 * hid
        self.lstm2 = nn.LSTM(
            input_size=hid * 2,
            hidden_size=hid * 2,
            batch_first=True,
        )

        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=hid * 2,
            num_heads=heads,
            dropout=drop,
            batch_first=True,
        )

        self.drop = nn.Dropout(drop)

        # Three-class classification
        self.fc = nn.Linear(hid * 2, n_class)

    def forward(self, x):

        h, _ = self.lstm1(x)
        h, _ = self.lstm2(h)

        attention, _ = self.attn(
            h, h, h,
            need_weights=False,
        )

        # Residual connection + last time step
        h = (attention + h)[:, -1, :]

        return self.fc(self.drop(h))


def predict_from_probs(probabilities, thresholds):

    buy_threshold, sell_threshold, margin = thresholds

    probabilities = np.asarray(probabilities)

    max_class = probabilities.argmax(axis=1)

    sorted_probs = np.sort(probabilities, axis=1)
    confidence_gap = sorted_probs[:, -1] - sorted_probs[:, -2]

    pred = np.ones(len(probabilities), dtype=np.int64)

    buy_mask = (
        (max_class == 0)
        & (probabilities[:, 0] > buy_threshold)
        & (confidence_gap >= margin)
    )

    sell_mask = (
        (max_class == 2)
        & (probabilities[:, 2] > sell_threshold)
        & (confidence_gap >= margin)
    )

    pred[buy_mask] = 0
    pred[sell_mask] = 2

    return pred


def apply_mode(probabilities, thresholds=None, mode="argmax"):


    if mode == "argmax":
        return np.asarray(probabilities).argmax(axis=1).astype(
            np.int64
        )

    if mode == "threshold":
        if thresholds is None:
            raise ValueError("Thresholds required")

        return predict_from_probs(probabilities, thresholds)

    raise ValueError(f"Unknown prediction mode: {mode}")
