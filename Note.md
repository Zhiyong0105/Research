#Apr 17
```
gcc -O3  -march=x86-64-v4 wave_4x3.c -fopenmp -lm -o wave_4x3
```
```
gcc -O3  -march=x86-64-v4 wave_4x3.c function.c -fopenmp -lm -o wave_4x3
```
```
sde-external-9.33.0-2024-01-07-lin/sde64 -skx -- 
```

```
git push -u origin main
```

```
OMP_NUM_THREADS=8 OMP_PLACES=0:8:2 sde-external-9.33.0-2024-01-07-lin/sde64 -skx -- ./wave_4x3
```

```
git pull orgin main
```
```
gcc -O3  -march=x86-64-v4 wave_auto.c function.c -fopenmp -lm -o wave_auto

```
```
OMP_NUM_THREADS=8 OMP_PLACES=0:8:2 sde-external-9.33.0-2024-01-07-lin/sde64 -skx -- ./wave_auto

```
```
echo "powersave" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```
```
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```
```
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_available_governors
```
```
cpupower frequency-info
```
```
watch -n 1 'cat /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq'
```
```
uname -a
```
```
echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```
```
cpupower frequency-set -u clock_freq
```
```
gcc -O3  -march=native wave_rev_auto.c -o wave_rev_auto apply_rev_avx.c -fopenmp -lm 
```
```
echo '1' | sudo tee  /sys/devices/system/cpu/intel_pstate/no_turbo
```
```
perf stat -e L2_RQSTS.REFERENCES,L2_RQSTS.RFO_HIT,L2_RQSTS.RFO_MISS,MEM_LOAD_RETIRED.L2_HIT,MEM_LOAD_RETIRED.L2_MISS,LLC-loads,LLC-load-misses
```
```
perf list
```
```
gcc -s / objdump -d
```
vanzeeらの実験比較
perf miss/op memory  

```
OMP_PROC_BIND=close OMP_PLACES=cores
```

3*3

n*y

## Sep 5
* ヒュージングサイズ`3X3`周りを試す



