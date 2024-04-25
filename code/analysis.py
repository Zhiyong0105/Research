import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

data = pd.read_csv("result.txt", sep=" ", header=None, names=["name","n","k","gflops","time","useless"])
data.to_csv("data.csv",index=False)
groupmean = data.groupby(["name","n","k"]).mean()
