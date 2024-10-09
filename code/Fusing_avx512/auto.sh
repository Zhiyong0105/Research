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
mxl=(2 3 4)
myl=(1 2 3)
nl=(1000 2000 3000 4000 5000)
kl=(96 192 384)
rm -f result.txt

# for x in "${mx[@]}"; do
#     for y in "${my[@]}"; do 
#         python3 Auto.py $x $y
#         gcc -O3 -march=x86-64-v4 wave_auto.c function.c -fopenmp -lm -o wave_auto
#         for n in "${ns[@]}"; do
#             for k in "${ks[@]}"; do
#                 OMP_NUM_THREADS=8 OMP_PLACES=0:8:2 sde-external-9.33.0-2024-01-07-lin/sde64 -skx -- ./wave_auto ${n} ${k} ${x} ${y} >> result.txt
#                 sleep 1
#             done
#         done
#     done
# done
# rm -f ../data/result_fusing.txt
# for x in "${mxl[@]}"; do
#     for y in "${myl[@]}"; do 
#         python3 Auto.py $x $y
#         gcc -O3 -march=x86-64-v4 fusing.c function.c -fopenmp -lm -o fusing
#         for n in "${nl[@]}"; do
#             for k in "${kl[@]}"; do
#                 OMP_NUM_THREADS=8  OMP_PROC_BIND=close OMP_PLACES=cores ./fusing ${n} ${k} ${x} ${y} >> ../data/result_fusing.txt
#                 # sleep 1
#             done
#         done
#     done
# done

mx3=(3)
my4=(4)
rm -f ../data/result_fusing_avx512.txt
for x in "${mxl[@]}"; do
    for y in "${myl[@]}"; do 
        python3 Auto.py $x $y
        gcc -O3 -march=x86-64-v4 fusing.c function.c -fopenmp -lm -o fusing
        for n in "${nl[@]}"; do
            for k in "${kl[@]}"; do
                OMP_NUM_THREADS=8  OMP_PROC_BIND=close OMP_PLACES=cores ./fusing ${n} ${k} ${x} ${y} >> ../data/result_fusing_avx512.txt
                # sleep 1
            done
        done
    done
done
