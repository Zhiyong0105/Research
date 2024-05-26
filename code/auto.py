import io
import sys


def apply_rev_auto_mv(my, mv):
    print("#include <stdio.h>")
    print("#include <stdlib.h>")
    print("#include <string.h>")
    print("#include <math.h>")
    print("#include <immintrin.h>")
    print("#include <pmmintrin.h>")
    print("#include \"apply_rev_avx.h\"")
    print("void apply_rev_avx_mv(int k, int m, int n, double *G, double *V, int ldv, int ldg, int my, int i, int mv)")
    print("{")

    str_init = "__m256d "
    for y in range(my+1):
        for v in range(mv):
            str_init += f" v{y}{v}, "

    str_init += "gamma, sigma, tmp;"
    print(str_init)
    
    #  loading for starting phase
    for y in range(my):
        for v in range(mv):
            # offset_y = f" +  ldv " if y == 1 else f" +  {y} * ldv"
            if y == 0:
                offset_y = ""
            elif y == 1:
                offset_y = " + ldv"
            else :
                offset_y = f" + {y} * ldv"    
            offset_v = f" + 4" if v !=0 else ""
            print(f"v{y}{v} = _mm256_loadu_pd(&V[i{offset_y}{offset_v}]);")
    
    # computing for givens rotation
   
    for y in range(my-1):
        for k in range(y+1):
            g = y - k
            
            offset_i = f" k " if k == 0 else f"(k + {k}) "
            if g == 0:
                offset_j = ""
            elif g == 1:
                offset_j = " + ldg "
            else :
                offset_j = f" {g} * ldg "
            print(f"    gamma = _mm256_broadcast_sd(&G[2 * {offset_i}{offset_j}]);")
            print(f"    sigma = _mm256_broadcast_sd(&G[2 * {offset_i}{offset_j} + 1]);")
            for v in range(mv):
                print(f" tmp = v{g}{v};")
                print(f" v{g}{v} = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v{g+1}{v}));")
                print(f" v{g+1}{v} = _mm256_sub_pd(_mm256_mul_pd(gamma, v{g+1}{v}), _mm256_mul_pd(sigma, tmp));")
    
    # Loop
    print(f"for (int g = {my-1}; g < n - 1; g++)")
    print("{")
    for v in range(mv):
        offset_v = f"i + (g + 1) * ldv + 4" if v == 0 else f"i + (g + 1) * ldv"
        print(f"v{my}{v }= _mm256_loadu_pd(&V[{offset_v}]);")
   
    for y in range(my):
        offset_zero_k = "k" if y==0 else f"(k + {y})"
        offset_zero_g = "g" if y == 0 else f"(g - {y})"

        print(f"gamma = _mm256_broadcast_sd(&G[2 * {offset_zero_k} + {offset_zero_g} * ldg]);")
        print(f"sigma = _mm256_broadcast_sd(&G[2 * {offset_zero_k} + {offset_zero_g} * ldg + 1]);")
        for v in range(mv):
            print(f"tmp = v{my-y-1}{v};")
            print(f" v{my-y-1}{v} = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v{my-y}{v}));")
            print(f" v{my-y}{v} = _mm256_sub_pd(_mm256_mul_pd(gamma, v{my-y}{v}), _mm256_mul_pd(sigma, tmp));")
        
    # store 
    for v in range(mv):
        print(f"_mm256_storeu_pd(&V[i + (g - {my-1}) * ldv], v{0}{v});")
    
    for y in range(my):
        for v in range(mv):
            print(f"v{y}{v}=v{y+1}{v};")
    print("}")
    
    # remaining
    for k in range(my-1):
        for g in range(my-k-1):
            gg = my - g - 2 
            ii = k + g + 1
            print(f"gamma = _mm256_broadcast_sd(&G[2 * (k + {ii}) + (n - {my-gg}) * ldg]);")
            print(f"sigma = _mm256_broadcast_sd(&G[2 * (k + {ii}) + (n - {my-gg}) * ldg + 1]);")
            for v in range(mv):
                print(f"tmp = v{gg}{v};")
                print(f" v{gg}{v} = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v{gg}{v}));")
                print(f" v{gg+1}{v} = _mm256_sub_pd(_mm256_mul_pd(gamma, v{gg+1}{v}), _mm256_mul_pd(sigma, tmp));")
    
    # store
    for y in range (my):
        for v in range(mv):
            offset_v = " ldv + 4" if v != 0 else " ldv"
            print(f"_mm256_storeu_pd(&V[i + (n - {my-y}) *{offset_v}], v{y}{v});")
        
    
            

            
            
            
            

    
                
                

                
            
        
            
            
        

    print("}")


original_stdout = sys.stdout
fake_stdout = io.StringIO()
sys.stdout = fake_stdout
args = sys.argv

# apply_rev_auto(int(args[1]), int(args[2]))
apply_rev_auto_mv(int(args[1]), int(args[2]))
# apply_rev_auto_mv(2, 2)

with open("apply_rev_avx.c", "w") as f:
    f.write(fake_stdout.getvalue())

sys.stdout = original_stdout
