import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

k_v = [24,96,192]
data = pd.read_csv("result.txt", sep=" ", header=None, names=["name","n","k","gflops","time","useless"])
data.to_csv("data.csv",index=False)
Groupmean_data = pd.read_csv("data.csv")
# print(Groupmean_data)
for k in k_v:
    plt.figure(figsize=(7, 5))
    for idx,name in enumerate(Groupmean_data["name"].unique()):
        #print(name)
        sub = Groupmean_data[(Groupmean_data["name"]==name)&(Groupmean_data["k"]==k)]
        # print(sub)
        # subset = Groupmean_data[(Groupmean_data["name"]==name)&(Groupmean_data["k"]==k)]
        # print(subset)
        plt.plot(sub['n'], sub['gflops'], label=name, color="red", linestyle="--",marker='o')
        plt.tight_layout()
        plt.savefig("G_husing_{}.pdf".format(k))
        plt.close()
