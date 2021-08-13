import numpy as np
import pandas as pd

data1 = pd.read_csv('./submission/submission(5).csv')
data2 = pd.read_csv('./submission/submission(6).csv')
data3 = pd.read_csv('./submission/submission(4).csv')

data4 = data1
data4["target"] = 0.40*data1["target"] + 0.40*data2["target"] + 0.20*data3["target"]

data3.to_csv("submission_ensemble.csv", index=False)