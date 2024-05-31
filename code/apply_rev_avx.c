#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <pmmintrin.h>
#include "apply_rev_avx.h" 
void apply_rev_avx_mv(int k, int m, int n, double *G, double *V, int ldv, int ldg, int i)
{
__m256d  v00,  v01,  v02,  v03,  v10,  v11,  v12,  v13,  v20,  v21,  v22,  v23, gamma, sigma, tmp;
v00 = _mm256_loadu_pd(&V[i]);
v01 = _mm256_loadu_pd(&V[i + 4 * 1]);
v02 = _mm256_loadu_pd(&V[i + 4 * 2]);
v03 = _mm256_loadu_pd(&V[i + 4 * 3]);
v10 = _mm256_loadu_pd(&V[i + ldv]);
v11 = _mm256_loadu_pd(&V[i + ldv + 4 * 1]);
v12 = _mm256_loadu_pd(&V[i + ldv + 4 * 2]);
v13 = _mm256_loadu_pd(&V[i + ldv + 4 * 3]);
    gamma = _mm256_broadcast_sd(&G[2 *  k ]);
    sigma = _mm256_broadcast_sd(&G[2 *  k  + 1]);
 tmp = v00;
 v00 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v10));
 v10 = _mm256_sub_pd(_mm256_mul_pd(gamma, v10), _mm256_mul_pd(sigma, tmp));
 tmp = v01;
 v01 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v11));
 v11 = _mm256_sub_pd(_mm256_mul_pd(gamma, v11), _mm256_mul_pd(sigma, tmp));
 tmp = v02;
 v02 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v12));
 v12 = _mm256_sub_pd(_mm256_mul_pd(gamma, v12), _mm256_mul_pd(sigma, tmp));
 tmp = v03;
 v03 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v13));
 v13 = _mm256_sub_pd(_mm256_mul_pd(gamma, v13), _mm256_mul_pd(sigma, tmp));
for (int g = 1; g < n - 1; g++)
{
v20= _mm256_loadu_pd(&V[i + (g + 1) * ldv ]);
v21= _mm256_loadu_pd(&V[i + (g + 1) * ldv + 4 * 1 ]);
v22= _mm256_loadu_pd(&V[i + (g + 1) * ldv + 4 * 2 ]);
v23= _mm256_loadu_pd(&V[i + (g + 1) * ldv + 4 * 3 ]);
gamma = _mm256_broadcast_sd(&G[2 * k + g * ldg]);
sigma = _mm256_broadcast_sd(&G[2 * k + g * ldg + 1]);
tmp = v10;
 v10 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v20));
 v20 = _mm256_sub_pd(_mm256_mul_pd(gamma, v20), _mm256_mul_pd(sigma, tmp));
tmp = v11;
 v11 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v21));
 v21 = _mm256_sub_pd(_mm256_mul_pd(gamma, v21), _mm256_mul_pd(sigma, tmp));
tmp = v12;
 v12 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v22));
 v22 = _mm256_sub_pd(_mm256_mul_pd(gamma, v22), _mm256_mul_pd(sigma, tmp));
tmp = v13;
 v13 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v23));
 v23 = _mm256_sub_pd(_mm256_mul_pd(gamma, v23), _mm256_mul_pd(sigma, tmp));
gamma = _mm256_broadcast_sd(&G[2 * (k + 1) + (g - 1) * ldg]);
sigma = _mm256_broadcast_sd(&G[2 * (k + 1) + (g - 1) * ldg + 1]);
tmp = v00;
 v00 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v10));
 v10 = _mm256_sub_pd(_mm256_mul_pd(gamma, v10), _mm256_mul_pd(sigma, tmp));
tmp = v01;
 v01 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v11));
 v11 = _mm256_sub_pd(_mm256_mul_pd(gamma, v11), _mm256_mul_pd(sigma, tmp));
tmp = v02;
 v02 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v12));
 v12 = _mm256_sub_pd(_mm256_mul_pd(gamma, v12), _mm256_mul_pd(sigma, tmp));
tmp = v03;
 v03 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v13));
 v13 = _mm256_sub_pd(_mm256_mul_pd(gamma, v13), _mm256_mul_pd(sigma, tmp));
_mm256_storeu_pd(&V[i + (g-1) * ldv ], v00);
_mm256_storeu_pd(&V[i + (g-1) * ldv + 4 * 1], v01);
_mm256_storeu_pd(&V[i + (g-1) * ldv + 4 * 2], v02);
_mm256_storeu_pd(&V[i + (g-1) * ldv + 4 * 3], v03);
v00=v10;
v01=v11;
v02=v12;
v03=v13;
v10=v20;
v11=v21;
v12=v22;
v13=v23;
}
gamma = _mm256_broadcast_sd(&G[2 * (k + 1) + (n - 2) * ldg]);
sigma = _mm256_broadcast_sd(&G[2 * (k + 1) + (n - 2) * ldg + 1]);
tmp = v00;
 v00 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v10));
 v10 = _mm256_sub_pd(_mm256_mul_pd(gamma, v10), _mm256_mul_pd(sigma, tmp));
tmp = v01;
 v01 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v11));
 v11 = _mm256_sub_pd(_mm256_mul_pd(gamma, v11), _mm256_mul_pd(sigma, tmp));
tmp = v02;
 v02 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v12));
 v12 = _mm256_sub_pd(_mm256_mul_pd(gamma, v12), _mm256_mul_pd(sigma, tmp));
tmp = v03;
 v03 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v13));
 v13 = _mm256_sub_pd(_mm256_mul_pd(gamma, v13), _mm256_mul_pd(sigma, tmp));
_mm256_storeu_pd(&V[i + (n - 2) * ldv], v00);
_mm256_storeu_pd(&V[i + (n - 2) * ldv + 4 * 1], v01);
_mm256_storeu_pd(&V[i + (n - 2) * ldv + 4 * 2], v02);
_mm256_storeu_pd(&V[i + (n - 2) * ldv + 4 * 3], v03);
_mm256_storeu_pd(&V[i + (n - 1) * ldv], v10);
_mm256_storeu_pd(&V[i + (n - 1) * ldv + 4 * 1], v11);
_mm256_storeu_pd(&V[i + (n - 1) * ldv + 4 * 2], v12);
_mm256_storeu_pd(&V[i + (n - 1) * ldv + 4 * 3], v13);
}
