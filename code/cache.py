import pandas as pd
import re
import csv


metrics = {
    "cpu_core/l2_request.all/": "l2_request.all",
    "cpu_core/l2_request.miss/": "l2_request.miss"
}


data = []


with open('perf_output.txt', 'r') as file:
    lines = file.readlines()

current_key = None


for line in lines:
    
    # print(f"Processing line: {line.strip()}")
    match = re.match(
        r" Performance counter stats for './wave_rev_auto (\d+) (\d+) (\d+) (\d+)':", line)
    if match:
        n, k, y, x = match.groups()
        current_key = {"n": n, "k": k, "y": y, "x": x, "l2_request.all": None, "l2_request.miss": None}
        # print(f"Matched parameters: n={n}, k={k}, y={y}, x={x}")
    else:
        if current_key:
            for metric_pattern, metric_name in metrics.items():
                if metric_pattern in line:
                   
                    value = re.search(r'(\d+[\d,]*)', line)
                    if value:
                        
                        current_key[metric_name] = int(
                            value.group(1).replace(',', ''))
                        # print( f"Matched {metric_name}: {current_key[metric_name]}")
    if current_key and all(current_key[metric_name] is not None for metric_name in metrics.values()):
        data.append(current_key)
        current_key = None
        
# for entry in data:
#     print(entry)


    


with open('perf_metrics.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=[
                            "n", "k", "y", "x"] + list(metrics.values()))
    writer.writeheader()
    for entry in data:
        writer.writerow(entry)



input_file = 'perf_metrics.csv'
output_file = 'perf_metrics_with_ratio.csv'

df = pd.read_csv(input_file)


df['miss_to_all_ratio'] = (df['l2_request.miss'] /
                           df['l2_request.all'] * 100).astype(int)


df.to_csv(output_file, index=False)

input_file = 'perf_metrics_with_ratio.csv'
df = pd.read_csv(input_file, sep=",", header=0, names=["n","k","y","x","l2_request.all","l2_request.miss","miss_to_all_ratio"])


Groupmean_cache_data = df.groupby(["n","k","y","x"]).mean().reset_index()

print(Groupmean_cache_data)


output_file = 'perf_metrics_with_ratio_mean.csv'
Groupmean_cache_data.to_csv(output_file, index=False)

print(f"Data with mean values saved to {output_file}")