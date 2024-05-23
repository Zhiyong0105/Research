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
    tmp = 0
    str_init = "__m256d "
    for i in range((my+1)*mv):
        str_init += "v{},".format(tmp)
        tmp += 1
    str_init += "gamma, sigma, tmp;"
    print(str_init)



    tmp = 0
    for i in range(my):
        for j in range(mv):
            offset_j = f" + {4 * j}" if 4 * j != 0 else ""
            if i != 0:
                if i==1:
                    offset_i = f" + ldv"
                else:
                    offset_i = f"{i} * ldv "
            else:
                offset_i=""
              
            print(f"    v{tmp} = _mm256_loadu_pd(&V[i{offset_i}{offset_j}]);")
            tmp += 1

    for k in range(my - 1):
        for y in range(k + 1):
            g = k - y
            offset_i = f"(k + {y}) " if y != 0 else "k "
            offset_j = f"+ {g} * ldg " if g != 0 else ""
            print(f"    gamma = _mm256_broadcast_sd(&G[2 * {offset_i}{offset_j}]);")
            print(f"    sigma = _mm256_broadcast_sd(&G[2 * {offset_i}{offset_j} + 1]);")

            for v in range(mv):
                print(f"    tmp = v{v};")
                print(f"    v{v} = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v{v + 2}));")
                print(f"    v{v + 2} = _mm256_sub_pd(_mm256_mul_pd(gamma, v{v + 2}), _mm256_mul_pd(sigma, tmp));")

            print("    for (int g = 1; g < n - 1; g++)")
            print("    {")
            print(f"        v{my * mv} = _mm256_loadu_pd(&V[i + (g + 1) * ldv]);")
            print(f"        v{my * mv + 1} = _mm256_loadu_pd(&V[i + (g + 1) * ldv + 4]);")

            tmp = mv
            for i in range(my):
                offset_i = f"(k + {i}) " if i != 0 else "k "
                offset_j = f" + (g - {i}) * ldg " if i != 0 else " + g * ldg"
                print(f"        gamma = _mm256_broadcast_sd(&G[2 * {offset_i}{offset_j}]);")
                print(f"        sigma = _mm256_broadcast_sd(&G[2 * {offset_i}{offset_j} + 1]);")
                
                for v in range(mv):
                    print(f"        tmp = v{tmp+v };")
                    print(f"        v{tmp+v } = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v{tmp+v+mv}));")
                    print(f"        v{tmp+v+mv} = _mm256_sub_pd(_mm256_mul_pd(gamma, v{tmp+v+mv}), _mm256_mul_pd(sigma, tmp));")
                    # tmp -= 1
                tmp -= 2

            print(f"        _mm256_storeu_pd(&V[i + (g - {my - 1}) * ldv], v0);")
            print(f"        _mm256_storeu_pd(&V[i + (g - {my - 1}) * ldv + 4], v1);")

            tmp = 0
            for i in range(my):
                for v in range(mv):
                    print(f"        v{tmp} = v{tmp + mv};")
                    tmp += 1
            print("    }")

    for k in range(my - 1):
        for j in range(my - k - 1):
            g = k - j
            i = k + j + 1
            print(f"    gamma = _mm256_broadcast_sd(&G[2 * (k + {i}) + (n - {my - g}) * ldg]);")
            print(f"    sigma = _mm256_broadcast_sd(&G[2 * (k + {i}) + (n - {my - g}) * ldg + 1]);")
            for v in range(mv):
                print(f"    tmp = v{v};")
                print(f"    v{v} = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v{v + mv}));")
                print(f"    v{v + mv} = _mm256_sub_pd(_mm256_mul_pd(gamma, v{v + mv}), _mm256_mul_pd(sigma, tmp));")
            tmp = 0

    for y in range(my):
        for v in range(mv):
            offset_i = f"ldv + 4" if v != 0 else "ldv"
            print(f"    _mm256_storeu_pd(&V[i + (n - {my - y}) * {offset_i}], v{tmp});")
            tmp += 1

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
