import pandas as pd
import matplotlib.pyplot as plt


cache_file = 'perf_metrics_with_ratio.csv'
cache_data = pd.read_csv(cache_file, sep=",", header=0, names=["n","k","y","x","l2_request.all","l2_request.miss","miss_to_all_ratio"])
Groupmean_cache_data = cache_data.groupby(["n","k","y","x"]).mean().reset_index()


result_file = 'result.txt'
data_rev = pd.read_csv(result_file, sep="\s+", header=None, names=["name","n","k","gflops","time","useless"])
Groupmean_data_rev_mean = data_rev.groupby(["name","n","k"]).median().reset_index()


unique_k_values_cache = Groupmean_cache_data['k'].unique()
unique_k_values_data_rev = Groupmean_data_rev_mean['k'].unique()
unique_k_values = set(unique_k_values_cache)


for k in unique_k_values:
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

   
    k_cache_data = Groupmean_cache_data[Groupmean_cache_data['k'] == k]
    unique_yx_combinations = k_cache_data[['y', 'x']].drop_duplicates()
    
    
    for _, row in unique_yx_combinations.iterrows():
        y, x = row['y'], row['x']
        subset = k_cache_data[(k_cache_data['y'] == y) & (k_cache_data['x'] == x)]
        ax1.plot(subset['n'], subset['miss_to_all_ratio'], label=f'{y}X{x}',linestyle='--', marker='o')
    
    ax1.set_title(f'Cache (L2): Ratio vs n for k={k}')
    ax1.set_xlabel('n')
    ax1.set_ylabel('miss_to_all_ratio')
    ax1.legend()
    ax1.grid(True)
    
   
    k_data_rev = Groupmean_data_rev_mean[Groupmean_data_rev_mean['k'] == k]
    unique_names = k_data_rev['name'].unique()
    
    
    for name in unique_names:
        subset = k_data_rev[k_data_rev['name'] == name]
        ax2.plot(subset['n'], subset['gflops'], label=f'{name}',linestyle='--', marker='o')
    
    ax2.set_title(f'GFLOPS vs n for k={k}')
    ax2.set_xlabel('n')
    ax2.set_ylabel('GFLOPS')
    ax2.legend()
    ax2.grid(True)
    
    
    output_image = f'picture/Cache_vs_GFLOPS_k_{k}.png'
    plt.savefig(output_image)
    plt.close()
    
   
