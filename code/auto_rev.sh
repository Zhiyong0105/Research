mv=(1 2 3 4)
my=(2 3)
ks=(24 48 96 192 384 768) 
ns=(1536  2304 3072 3840)
nd=(1000  2000  3000  4000 5000 6000)
rm -f result.txt
# for ((i=0; i<4; i++)); do
#     for ((j=0; j<5-i; j++)); do
#         x="${mv[i]}"
#         y="${my[j]}"
#         python3 auto.py "$y" "$x"
#         gcc -O3 -march=native wave_rev_auto.c -o wave_rev_auto apply_rev_avx.c -fopenmp -lm
#         for n in "${ns[@]}"; do
#             for k in "${ks[@]}"; do
#                 OMP_NUM_THREADS=8 OMP_PLACES=0:8:2 ./wave_rev_auto "${n}" "${k}" "${y}" "${x}" >> result.txt
#             done
#         done
        
#     done
# done

for ((i=0; i<4; i++)); do
    for ((j=0; j<2; j++)); do
        x="${mv[i]}"
        y="${my[j]}"
        if [ "$x" -eq 4 ] && [ "$y" -eq 3 ]; then
            outer_break=true
            break
        fi
        python3 auto.py "$y" "$x"
        gcc -O3 -march=native wave_rev_auto.c -o wave_rev_auto apply_rev_avx.c -fopenmp -lm
        
        for n in "${nd[@]}"; do
            for k in "${ks[@]}"; do
                OMP_NUM_THREADS=8 OMP_PLACES=0:8:2 ./wave_rev_auto "${n}" "${k}" "${y}" "${x}" >> result.txt
                done
            done
        
    done
    
done


