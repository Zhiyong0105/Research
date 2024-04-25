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
gcc -O3  -march=x86-64-v4 wave_auto.c function.c -fopenmp -lm -o wave_4x3

```