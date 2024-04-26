import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

k_v = [24,96,192]
data = pd.read_csv("data/result_sleep.txt", sep=" ", header=None, names=["name","n","k","gflops","time","useless"])
data_op = pd.read_csv("data/result_sleep_op.txt", sep=" ", header=None, names=["name","n","k","gflops","time","useless"])

Groupmean_data_mean= data.groupby(["name","n","k"]).mean()
Groupmean_data_mean = Groupmean_data_mean.reset_index()

Groupmean_data_op_mean= data_op.groupby(["name","n","k"]).mean()
Groupmean_data_op_mean = Groupmean_data_mean.reset_index()

for k in k_v:
    plt.figure(figsize=(8, 6))
    for name in Groupmean_data_mean["name"].unique():
        sub = Groupmean_data_mean[(Groupmean_data_mean["name"] == name) & (Groupmean_data_mean['k'] == k)]
        if not sub.empty:
            if name == '4X3':
                plt.plot(sub['n'], sub['gflops'], label=name, color='red', linestyle='-', marker='o')
            else:
                plt.plot(sub['n'], sub['gflops'], label=name, linestyle='--', marker='o')
    plt.legend()
    plt.xlabel('n')
    plt.ylabel('gflops')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("picture/G_husing_sleep{}.pdf".format(k))
    plt.close()

for k in k_v:
    plt.figure(figsize=(8, 6))
    for name in Groupmean_data_op_mean["name"].unique():
        sub = Groupmean_data_op_mean[(Groupmean_data_op_mean["name"] == name) & (Groupmean_data_op_mean['k'] == k)]
        if not sub.empty:
            if name == '4X3':
                plt.plot(sub['n'], sub['gflops'], label=name, color='red', linestyle='-', marker='o')
            else:
                plt.plot(sub['n'], sub['gflops'], label=name, linestyle='--', marker='o')
    plt.legend()
    plt.xlabel('n')
    plt.ylabel('gflops')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("picture/G_husing_sleep_op{}.pdf".format(k))
    plt.close()