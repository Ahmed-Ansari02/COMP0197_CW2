import torch
import pandas as pd
preds = pd.read_csv("results/bayesian_lstm/predictions.csv")
print(preds[preds["y_true"] == 0]["mu_mean"].describe())
print(preds[preds["y_true"] == 0]["lo_80"].describe())