import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
data_avx256_my_3_amd = pd.read_csv('data/result_avx256_my_3_amd.txt', sep=" ", header=None, names=["name", "n", "k", "gflops", "time", "useless"])
data_rev_fusing_avx256 = pd.read_csv('data/result_rev_fusing_avx256.txt', sep=" ", header=None, names=["name", "n", "k", "gflops", "time", "useless"])
data_avx512_my_3_lowGHz = pd.read_csv('data/result_avx512_my_3_lowGHz.txt', sep=" ", header=None, names=["name", "n", "k", "gflops", "time", "useless"])
# 筛选出name为"3X3"的数据
data_avx256_my_3_amd = data_avx256_my_3_amd[data_avx256_my_3_amd["name"] == "3X3"]
data_rev_fusing_avx256 = data_rev_fusing_avx256[data_rev_fusing_avx256["name"] == "3X3"]
data_avx512_my_3_lowGHz = data_avx512_my_3_lowGHz[data_avx512_my_3_lowGHz["name"] == "3X3"]

Groupby_data_avx256_my_3_amd_mean = data_avx256_my_3_amd.groupby(["name","n","k"]).mean()
Groupby_data_avx256_my_3_amd_mean = Groupby_data_avx256_my_3_amd_mean.reset_index()

Groupby_data_rev_fusing_avx256_mean = data_rev_fusing_avx256.groupby(["name","n","k"]).mean()
Groupby_data_rev_fusing_avx256_mean = Groupby_data_rev_fusing_avx256_mean.reset_index()

Groupby_data_avx512_my_3_lowGHz_mean = data_avx512_my_3_lowGHz.groupby(["name","n","k"]).mean()
Groupby_data_avx512_my_3_lowGHz_mean = Groupby_data_avx512_my_3_lowGHz_mean.reset_index()
# 获取所有唯一的k值
unique_k_values = sorted(data_avx256_my_3_amd['k'].unique())

# 根据每个k值绘制图像
for k in unique_k_values:
    plt.figure(figsize=(8, 6))
    
    # 筛选k对应的数据
    sub_my_3_amd = Groupby_data_avx256_my_3_amd_mean[Groupby_data_avx256_my_3_amd_mean['k'] == k]
    sub_rev_fusing_avx256 = Groupby_data_rev_fusing_avx256_mean[Groupby_data_rev_fusing_avx256_mean['k'] == k]
    sub_avx512_my_3_lowGHz = Groupby_data_avx512_my_3_lowGHz_mean[Groupby_data_avx512_my_3_lowGHz_mean['k'] == k]
    
    # 绘制每个数据集
    if not sub_my_3_amd.empty:
        plt.plot(sub_my_3_amd['n'], sub_my_3_amd['gflops'], label="AVX256 my 3 amd", color='blue', linestyle='-', marker='o')
    if not sub_rev_fusing_avx256.empty:
        plt.plot(sub_rev_fusing_avx256['n'], sub_rev_fusing_avx256['gflops'], label="Rev Fusing AVX256", color='green', linestyle='-', marker='o')
    if not sub_avx512_my_3_lowGHz.empty:
        plt.plot(sub_avx512_my_3_lowGHz['n'], sub_avx512_my_3_lowGHz['gflops'], label="AVX512 my 3 low GHz", color='red', linestyle='-', marker='o')
    
    # 图形设置
    plt.legend()
    plt.title(f'Performance comparison for k={k}')
    plt.xlabel('n')
    plt.ylabel('gflops')
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    # 保存图片
    plt.savefig(f"picture/combined_3X3_gflops_k_{k}.png")
    plt.close()


