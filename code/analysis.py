import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

k_v = [24,48,96,192,384,768]
data = pd.read_csv("data/result_sleep.txt", sep=" ", header=None, names=["name","n","k","gflops","time","useless"])
data_op = pd.read_csv("data/result_sleep_op.txt", sep=" ", header=None, names=["name","n","k","gflops","time","useless"])
data_rev = pd.read_csv("result.txt", sep=" ", header=None, names=["name","n","k","gflops","time","useless"])
data_rev_avx512 = pd.read_csv("result_avx512.txt", sep=" ", header=None, names=["name","n","k","gflops","time","useless"])
data_rev_avx512_seq =pd.read_csv("data/result_avx512_seq.txt", sep=" ", header=None, names=["name","n","k","gflops","time","useless"])

Groupmean_data_mean= data.groupby(["name","n","k"]).mean()
Groupmean_data_mean = Groupmean_data_mean.reset_index()

Groupmean_data_rev_mean=data_rev.groupby(["name","n","k"]).median()

Groupmean_data_rev_mean= Groupmean_data_rev_mean.reset_index()

Groupmean_data_rev_avx512_mean=data_rev_avx512.groupby(["name","n","k"]).mean()
Groupmean_data_rev_avx512_mean= Groupmean_data_rev_avx512_mean.reset_index()

Groupmean_data_rev_avx512_seq_mean=data_rev_avx512_seq.groupby(["name","n","k"]).mean()
Groupmean_data_rev_avx512_seq_mean= Groupmean_data_rev_avx512_seq_mean.reset_index()

Groupmean_data_op_mean= data_op.groupby(["name","n","k"]).mean()
Groupmean_data_op_mean = Groupmean_data_op_mean.reset_index()

theoretical_performance = 201
Groupmean_data_rev_mean['gflops_percentage'] = (Groupmean_data_rev_mean['gflops'] / theoretical_performance) * 100

theoretical_performance_avx512 = 384
Groupmean_data_rev_avx512_seq_mean['gflops_percentage']= (Groupmean_data_rev_avx512_seq_mean['gflops'] / theoretical_performance_avx512) * 100


k_name = Groupmean_data_rev_mean['k'].unique()
k_name = set(k_name)

k_512_name = Groupmean_data_rev_avx512_seq_mean['k'].unique()
k_512_name = set(k_512_name)



for k in k_512_name:
    plt.figure(figsize=(8, 6))
    for name in Groupmean_data_rev_avx512_seq_mean["name"].unique():
        sub = Groupmean_data_rev_avx512_seq_mean[(Groupmean_data_rev_avx512_seq_mean["name"] == name) & (Groupmean_data_rev_avx512_seq_mean['k'] == k)]
        if not sub.empty:
            if name == '3X3':
                plt.plot(sub['n'], sub['gflops'], label=name, color='red', linestyle='-', marker='o')
            else:
                plt.plot(sub['n'], sub['gflops'], label=name, linestyle='--', marker='o')
    plt.legend()
    plt.xlabel('n')
    plt.ylabel('gflops')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("picture/G_husing_rev_avx512_op_{}.png".format(k))
    plt.close()
# for k in k_v:
#     plt.figure(figsize=(8, 6))
#     for name in Groupmean_data_rev_avx512_seq_mean["name"].unique():
#         sub = Groupmean_data_rev_avx512_seq_mean[(Groupmean_data_rev_avx512_seq_mean["name"] == name) & (Groupmean_data_rev_avx512_seq_mean['k'] == k)]
#         if not sub.empty:
#             if name == '3X3':
#                 plt.plot(sub['n'], sub['gflops_percentage'], label=name, color='red', linestyle='-', marker='o')
#             else:
#                 plt.plot(sub['n'], sub['gflops_percentage'], label=name, linestyle='--', marker='o')
#     plt.legend()
#     plt.xlabel('n')
#     plt.ylabel('gflops (%)')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig("picture/G_husing_rev_avx512_seq_lar_per{}.png".format(k))
#     plt.close()
    
# for k in k_v:
#     plt.figure(figsize=(8, 6))
#     for name in Groupmean_data_rev_mean["name"].unique():
#         sub = Groupmean_data_rev_mean[(Groupmean_data_rev_mean["name"] == name) & (Groupmean_data_rev_mean['k'] == k)]
#         if not sub.empty:
#             if name == '3X3':
#                 plt.plot(sub['n'], sub['gflops_percentage'], label=name, color='red', linestyle='-', marker='o')
#             else:
#                 plt.plot(sub['n'], sub['gflops_percentage'], label=name, linestyle='--', marker='o')
#     plt.legend()
#     plt.xlabel('n')
#     plt.ylabel('gflops (%)')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig("picture/G_husing_rev_new_per{}.png".format(k))
#     plt.close()
for k in k_name:
    plt.figure(figsize=(8, 6))
    for name in Groupmean_data_rev_mean["name"].unique():
        sub = Groupmean_data_rev_mean[(Groupmean_data_rev_mean["name"] == name) & (Groupmean_data_rev_mean['k'] == k)]
        if not sub.empty:
            if name == '3X3':
                plt.plot(sub['n'], sub['gflops'], label=name, color='red', linestyle='-', marker='o')
            else:
                plt.plot(sub['n'], sub['gflops'], label=name, linestyle='--', marker='o')
    plt.legend()
    plt.xlabel('n')
    plt.ylabel('gflops (%)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("picture/G_husing_rev_new_{}.png".format(k))
    plt.close()
    
# for k in k_v:
#     plt.figure(figsize=(8, 6))
#     for name in Groupmean_data_mean["name"].unique():
#         sub = Groupmean_data_mean[(Groupmean_data_mean["name"] == name) & (Groupmean_data_mean['k'] == k)]
#         if not sub.empty:
#             if name == '4X3':
#                 plt.plot(sub['n'], sub['gflops'], label=name, color='red', linestyle='-', marker='o')
#             else:
#                 plt.plot(sub['n'], sub['gflops'], label=name, linestyle='--', marker='o')
#     plt.legend()
#     plt.xlabel('n')
#     plt.ylabel('gflops')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig("picture/G_husing_sleep{}.pdf".format(k))
#     plt.close()

# for k in k_v:
#     plt.figure(figsize=(8, 6))
#     for name in Groupmean_data_op_mean["name"].unique():
#         sub = Groupmean_data_op_mean[(Groupmean_data_op_mean["name"] == name) & (Groupmean_data_op_mean['k'] == k)]
#         if not sub.empty:
#             if name == '4X3':
#                 plt.plot(sub['n'], sub['gflops'], label=name, color='red', linestyle='-', marker='o')
#             else:
#                 plt.plot(sub['n'], sub['gflops'], label=name, linestyle='--', marker='o')
#     plt.legend()
#     plt.xlabel('n')
#     plt.ylabel('gflops')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig("picture/G_husing_sleep_op{}.pdf".format(k))
#     plt.close()