import io
import sys
def apply_rev_auto(my,mv=1):
    print("#include <stdio.h>")
    print("#include <stdlib.h>")
    print("#include <string.h>")
    print("#include <math.h>")
    print("#include <immintrin.h>")
    print("#include <pmmintrin.h>")
    print("#include \"apply_rev_avx.h\" ")
    print("void apply_rev_avx(int k, int m, int n, double *G, double *V, int ldv, int ldg,int my,int i) ")
    print("{")
    str_startv = "__m256d  v0"
    for k in range(1,my+1):
        str_startv += ",v{} ".format(k)
    str_startv +=",gamma,sigma,tmp;"
    print(str_startv)
    for i in range(my):
        print("v{0} = _mm256_loadu_pd(&V[i+{1}*ldv]);".format(i,i))
    for k in range(my-1):
        for y in range(k+1):
        # print(y,k)
            gg = k-y
            ii = y
            # print(gg,ii)
            print("gamma = _mm256_broadcast_sd(&G[2 * (k+{0})+{1}*ldg]);".format(ii,gg))
            print("sigma = _mm256_broadcast_sd(&G[2 * (k+{0})+{1}*ldg+1]);".format(ii,gg))
            print("tmp = v{};".format(gg))
            print("v{0} = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v{1}));".format(gg,gg+1))
            print("v{0} = _mm256_sub_pd(_mm256_mul_pd(gamma, v{1}), _mm256_mul_pd(sigma, tmp));".format(gg+1,gg+1))   

        # print(gg,ii)
    print("for (int g = {}; g < n - 1; g++)".format(my-1))
    print("{")
    print("v{} = _mm256_loadu_pd(&V[i + (g + 1) * ldv]);".format(my))
    for i in range(my):
            print("gamma = _mm256_broadcast_sd(&G[2 * (k+{0})+(g-{1})*ldg]);".format(i,i))
            print("sigma = _mm256_broadcast_sd(&G[2 * (k+{0})+(g-{1})*ldg+1]);".format(i,i))
            print("tmp = v{};".format(my-i-1))
            print("v{0} = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v{1}));".format(my-i-1,my-i))
            print("v{0} = _mm256_sub_pd(_mm256_mul_pd(gamma, v{1}), _mm256_mul_pd(sigma, tmp));".format(my-i,my-i))  
    print("_mm256_storeu_pd(&V[i + (g - {}) * ldv], v0);".format(my-1))
    for i in range(my):
        print("v{0}=v{1};".format(i,i+1))
    print("}")  

    for k in range(my-1):
        for j in range(my-k-1):
            gg  = -j +my -2
            ii = j+k+1
            print("gamma = _mm256_broadcast_sd(&G[2 * (k+{0})+(n-{1})*ldg]);".format(ii,my-gg))
            print("sigma = _mm256_broadcast_sd(&G[2 * (k+{0})+(n-{1})*ldg+1]);".format(ii,my-gg))
            print("tmp = v{};".format(gg))
            print("v{0} = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v{1}));".format(gg,gg+1))
            print("v{0} = _mm256_sub_pd(_mm256_mul_pd(gamma, v{1}), _mm256_mul_pd(sigma, tmp));".format(gg+1,gg+1))  
    for k in range(my):
        print("_mm256_storeu_pd(&V[i + (n - {0}) * ldv], v{1});".format(my-k,k)) 
    print("}")


original_stdout = sys.stdout
fake_stdout = io.StringIO()
sys.stdout = fake_stdout
args = sys.argv





apply_rev_auto(int(args[1]), int(args[2]))


with open("apply_rev_avx.c", "w") as f:
    f.write(fake_stdout.getvalue())


sys.stdout = original_stdout        
    
    

        



