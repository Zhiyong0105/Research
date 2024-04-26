ns=(
    "1000"
    "2000"
    "3000"
    "4000"

)
ks=(
    "24"
    "96"
    "192"
)

mx=(

    "2"
    "3"
    "4"
)
my=(
    "1"
    "2"
    "3"
)

rm -f result.txt

for x in "${mx[@]}"; do
    for y in "${my[@]}"; do 
        python3 Auto.py $x $y
        gcc -O3 -march=x86-64-v4 wave_auto.c function.c -fopenmp -lm -o wave_auto
        for n in "${ns[@]}"; do
            for k in "${ks[@]}"; do
                OMP_NUM_THREADS=8 OMP_PLACES=0:8:2 sde-external-9.33.0-2024-01-07-lin/sde64 -skx -- ./wave_auto ${n} ${k} ${x} ${y} >> result.txt
                sleep 1
            done
        done
    done
done
