mv=(1 2 3 4)
my=(2 3)
ks=(96 192 384 ) 
ns=(1536  2304 3072 3840)
nd=(1000 2000 3000 4000 5000)
ng=(1440 2880  4320 5760)
myxmv=(6 4 4 5 3 6 3 3 3 2 2 3)


three_mv=(3 1 3 2 3 3 3 4 3 5 3 6 3 7)

my_3=(1 3 2 3 3 3 4 3 6 3)
n_my_3=(960 1920 2880 3840 4800)
n_my_3_fusing=(1000 2000 3000 4000 5000)

three_3=(3 3)
n_3_3=(960 1920 2880 3840 4800)
three_6=(3 6)
n_3_6=(1152 1536 2304 2688 3072 3840 4608)

# n_100=(1000 11000 1200 1300 1400 1500 1600 1700 1800 1900 2000 2100 22)
n_200=($(seq 1000 200 5000))


# rm -f result.txt
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

# for ((i=0; i<4; i++)); do
#     for ((j=0; j<2; j++)); do
#         x="${mv[i]}"
#         y="${my[j]}"
#         if [ "$x" -eq 4 ] && [ "$y" -eq 3 ]; then
#             outer_break=true
#             break
#         fi
#         python3 auto.py "$y" "$x"
#         gcc -O3 -march=native wave_rev_auto.c -o wave_rev_auto apply_rev_avx.c -fopenmp -lm
        
#         for n in "${ns[@]}"; do
#             for k in "${ks[@]}"; do
#                 OMP_NUM_THREADS=8 OMP_PLACES=0:8:2 ./wave_rev_auto "${n}" "${k}" "${y}" "${x}" >> result.txt
#                 done
#             done
        
#     done
    
# done
# rm -f result_avx512.txt
# for ((i=0; i<4; i++)); do
#     for ((j=0; j<2; j++)); do
#         x="${mv[i]}"
#         y="${my[j]}"
#         if [ "$x" -eq 4 ] && [ "$y" -eq 3 ]; then
#             outer_break=true
#             break
#         fi
#         python3 auto_avx512.py "$y" "$x"
#         gcc -O3 -march=x86-64-v4 wave_rev_auto.c -o wave_rev_auto apply_rev_avx.c -fopenmp -lm
#         for n in "${nd[@]}"; do
#             for k in "${ks[@]}"; do
#                 OMP_NUM_THREADS=8 OMP_PLACES=0:8:2 ./wave_rev_auto "${n}" "${k}" "${y}" "${x}" >> result_avx512.txt
#                 done
#             done
        
#     done
    
# done
# rm -f result_avx512.txt
# for ((i=0;i<12;i+=2));do
#     y="${myxmv[i]}"
#     x="${myxmv[i+1]}"
#     python3 auto_avx512.py "$y" "$x"
# #     gcc -O3 -march=x86-64-v4 wave_rev_auto.c -o wave_rev_auto apply_rev_avx.c -fopenmp -lm
#       gcc -O3 -march=x86-64-v4  wave_rev_auto.c -o wave_rev_auto apply_rev_avx.c -fopenmp   -lm

#     for n in "${ng[@]}"; do
#         for k in "${ks[@]}"; do
#                 # OMP_NUM_THREADS=8 OMP_PLACES=0:8:2 
#                 # OMP_PROC_BIND=close OMP_PLACES=cores 
#                OMP_NUM_THREADS=8  OMP_PROC_BIND=close OMP_PLACES=cores ./wave_rev_auto "${n}" "${k}" "${y}" "${x}" >> result_avx512.txt
#                 done
#         done
# done

# rm -f result_avx512_mv_3.txt
# for ((i=0;i<10;i+=2));do
#     y="${my_3[i]}"
#     x="${my_3[i+1]}"
#     python3 auto_avx512.py "$y" "$x"
#     # gcc -O3 -march=x86-64-v4 wave_rev_auto.c -o wave_rev_auto apply_rev_avx.c -fopenmp -lm
#       gcc-13 -O3 -march=znver4  wave_rev_auto.c -o wave_rev_auto apply_rev_avx.c -fopenmp   -lm
#     for n in "${n_my_3_fusing[@]}"; do
#         for k in "${ks[@]}"; do
#                 # OMP_NUM_THREADS=8 OMP_PLACES=0:8:2 
#                 # OMP_PROC_BIND=close OMP_PLACES=cores 
#                OMP_NUM_THREADS=8  OMP_PROC_BIND=close OMP_PLACES=cores ./wave_rev_auto "${n}" "${k}" "${y}" "${x}" >> result_avx512_mv_3.txt
#                 done
#         done
# done

# rm -f result_avx512_my_3.txt
# for ((i=0;i<14;i+=2));do
#     y="${three_mv[i]}"
#     x="${three_mv[i+1]}"
#     python3 auto_avx512.py "$y" "$x"
#     # gcc -O3 -march=x86-64-v4 wave_rev_auto.c -o wave_rev_auto apply_rev_avx.c -fopenmp -lm
#     gcc-13 -O3 -march=znver4   wave_rev_auto.c -o wave_rev_auto apply_rev_avx.c -fopenmp   -lm

#     for n in "${nd[@]}"; do
#         for k in "${ks[@]}"; do
#                 # OMP_NUM_THREADS=8 OMP_PLACES=0:8:2 
#                 # OMP_PROC_BIND=close OMP_PLACES=cores 
#                OMP_NUM_THREADS=8  OMP_PROC_BIND=close OMP_PLACES=cores ./wave_rev_auto "${n}" "${k}" "${y}" "${x}" >> result_avx512_my_3.txt
#                 done
#         done
# done


# rm -f result_avx256_my_3_amd.txt
# for ((i=0;i<14;i+=2));do
#     y="${three_mv[i]}"
#     x="${three_mv[i+1]}"
#     python3 auto_avx512.py "$y" "$x"
#     # gcc -O3 -march=x86-64-v4 wave_rev_auto.c -o wave_rev_auto apply_rev_avx.c -fopenmp -lm
#     gcc-13 -O3 -march=znver4   wave_rev_auto.c -o wave_rev_auto apply_rev_avx.c -fopenmp   -lm

#     for n in "${nd[@]}"; do
#         for k in "${ks[@]}"; do
#                 # OMP_NUM_THREADS=8 OMP_PLACES=0:8:2 
#                 # OMP_PROC_BIND=close OMP_PLACES=cores 
#                OMP_NUM_THREADS=8  OMP_PROC_BIND=close OMP_PLACES=cores ./wave_rev_auto "${n}" "${k}" "${y}" "${x}" >> result_avx256_my_3_amd.txt
#                 done
#         done
# done
# rm -f result_avx512_my_3_lowGHz.txt
# for ((i=0;i<14;i+=2));do
#     y="${three_mv[i]}"
#     x="${three_mv[i+1]}"
#     python3 auto_avx512.py "$y" "$x"
#     # gcc -O3 -march=x86-64-v4 wave_rev_auto.c -o wave_rev_auto apply_rev_avx.c -fopenmp -lm
#     gcc-13 -O3 -march=znver4   wave_rev_auto.c -o wave_rev_auto apply_rev_avx.c -fopenmp   -lm

#     for n in "${nd[@]}"; do
#         for k in "${ks[@]}"; do
#                 # OMP_NUM_THREADS=8 OMP_PLACES=0:8:2 
#                 # OMP_PROC_BIND=close OMP_PLACES=cores 
#                OMP_NUM_THREADS=8  OMP_PROC_BIND=close OMP_PLACES=cores ./wave_rev_auto "${n}" "${k}" "${y}" "${x}" >> result_avx512_my_3_lowGHz.txt
#                 done
#         done
# done

# rm -f result_avx512_my_3.txt
# for ((i=0; i<14; i+=2)); do
#     y="${three_mv[i]}"
#     x="${three_mv[i+1]}"
#     python3 auto_avx512.py "$y" "$x"
    
#     gcc -O3 -march=x86-64-v4 wave_rev_auto.c -o wave_rev_auto apply_rev_avx.c -fopenmp -lm -lnuma

#     for n in "${nd[@]}"; do
#         for k in "${ks[@]}"; do
            
#             export OMP_NUM_THREADS=8
#             export OMP_PROC_BIND=close
#             export OMP_PLACES=cores
            
            
#             numactl --cpunodebind=0 --membind=0 ./wave_rev_auto "${n}" "${k}" "${y}" "${x}" >> result_avx512_my_3.txt
#         done
#     done
# done


# rm -f result_avx512_3_3.txt
# for ((i=0;i<2;i+=2));do
#     y="${three_3[i]}"
#     x="${three_3[i+1]}"
#     python3 auto_avx512.py "$y" "$x"
# #     gcc -O3 -march=x86-64-v4 wave_rev_auto.c -o wave_rev_auto apply_rev_avx.c -fopenmp -lm
#       gcc -O3 -march=x86-64-v4  wave_rev_auto.c -o wave_rev_auto apply_rev_avx.c -fopenmp   -lm

#     for n in "${nd[@]}"; do
#         for k in "${ks[@]}"; do
#                 # OMP_NUM_THREADS=8 OMP_PLACES=0:8:2 
#                 # OMP_PROC_BIND=close OMP_PLACES=cores 
#                OMP_NUM_THREADS=8  OMP_PROC_BIND=close OMP_PLACES=cores ./wave_rev_auto "${n}" "${k}" "${y}" "${x}" >> result_avx512_3_3.txt
#                 done
#         done
# done

# rm -f result_avx512_3_6.txt
# for ((i=0;i<2;i+=2));do
#     y="${three_6[i]}"
#     x="${three_6[i+1]}"
#     python3 auto_avx512.py "$y" "$x"
# #     gcc -O3 -march=x86-64-v4 wave_rev_auto.c -o wave_rev_auto apply_rev_avx.c -fopenmp -lm
#       gcc -O3 -march=x86-64-v4  wave_rev_auto.c -o wave_rev_auto apply_rev_avx.c -fopenmp   -lm

#     for n in "${nd[@]}"; do
#         for k in "${ks[@]}"; do
#                 # OMP_NUM_THREADS=8 OMP_PLACES=0:8:2 
#                 # OMP_PROC_BIND=close OMP_PLACES=cores 
#                OMP_NUM_THREADS=8  OMP_PROC_BIND=close OMP_PLACES=cores ./wave_rev_auto "${n}" "${k}" "${y}" "${x}" >> result_avx512_3_6.txt
#                 done
#         done
# done

# for ((i=0; i<4; i++)); do
#     for ((j=0; j<2; j++)); do
#         x="${mv[i]}"
#         y="${my[j]}"
#         if [ "$x" -eq 4 ] && [ "$y" -eq 3 ]; then
#             outer_break=true
#             break
#         fi
#         python3 auto_avx512.py "$y" "$x"
#         gcc -O3 -march=x86-64-v4 wave_rev_auto.c -o wave_rev_auto apply_rev_avx.c -fopenmp -lm
#         for n in "${nd[@]}"; do
#             for k in "${ks[@]}"; do
#                 OMP_NUM_THREADS=8 OMP_PLACES=0:8:2 ./wave_rev_auto "${n}" "${k}" "${y}" "${x}" >> result_avx512.txt
#                 done
#             done
        
#     done
    
# done

# rm -f result_avx512_fma_my_3.txt
# for ((i=0;i<14;i+=2));do
#     y="${three_mv[i]}"
#     x="${three_mv[i+1]}"
#     python3 auto_avx512.py "$y" "$x"
#     # gcc -O3 -march=x86-64-v4 wave_rev_auto.c -o wave_rev_auto apply_rev_avx.c -fopenmp -lm
#     gcc-13 -O3 -march=znver4   wave_rev_auto.c -o wave_rev_auto apply_rev_avx.c -fopenmp   -lm

#     for n in "${n_200[@]}"; do
#         for k in "${ks[@]}"; do
#                 # OMP_NUM_THREADS=8 OMP_PLACES=0:8:2 
#                 # OMP_PROC_BIND=close OMP_PLACES=cores 
#                OMP_NUM_THREADS=8  OMP_PROC_BIND=close OMP_PLACES=cores ./wave_rev_auto "${n}" "${k}" "${y}" "${x}" >> result_avx512_fma_my_3.txt
#                 done
#                 sleep 1
#         done
        
# done

rm -f result_analysis_my_3.txt
for ((i=0; i<14; i+=2)); do
    y="${three_mv[i]}"
    x="${three_mv[i+1]}"
    python3 auto_avx512.py "$y" "$x"
    gcc-13 -O3 -march=znver4 wave_rev_auto.c -o wave_rev_auto apply_rev_avx.c -fopenmp -lm

    for n in "${nd[@]}"; do
        for k in "${ks[@]}"; do
            OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores \
            perf stat -e cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses \
            -o perf_output.txt --append ./wave_rev_auto "${n}" "${k}" "${y}" "${x}" > wave_output.txt
            
            echo "n=${n}, k=${k}, y=${y}, x=${x}" >> result_analysis_my_3.txt
            grep "GFLOPS" wave_output.txt >> result_analysis_my_3.txt
            cat perf_output.txt >> result_analysis_my_3.txt

            rm -f wave_output.txt perf_output.txt
            sleep 1
        done
    done
done




