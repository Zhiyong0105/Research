#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <pmmintrin.h>
#include "apply_rev_avx.h"
void apply_rev_avx_mv(int k, int m, int n, double *G, double *V, int ldv, int ldg, int i)
{
__m256d  v00,  v01,  v10,  v11,  v20,  v21,  v30,  v31, gamma, sigma, tmp;
v00 = _mm256_loadu_pd(&V[i]);
v01 = _mm256_loadu_pd(&V[i + 4]);
v10 = _mm256_loadu_pd(&V[i + ldv]);
v11 = _mm256_loadu_pd(&V[i + ldv + 4]);
v20 = _mm256_loadu_pd(&V[i + 2 * ldv]);
v21 = _mm256_loadu_pd(&V[i + 2 * ldv + 4]);
    gamma = _mm256_broadcast_sd(&G[2 *  k ]);
    sigma = _mm256_broadcast_sd(&G[2 *  k  + 1]);
 tmp = v00;
 v00 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v10));
 v10 = _mm256_sub_pd(_mm256_mul_pd(gamma, v10), _mm256_mul_pd(sigma, tmp));
 tmp = v01;
 v01 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v11));
 v11 = _mm256_sub_pd(_mm256_mul_pd(gamma, v11), _mm256_mul_pd(sigma, tmp));
    gamma = _mm256_broadcast_sd(&G[2 *  k  + ldg ]);
    sigma = _mm256_broadcast_sd(&G[2 *  k  + ldg  + 1]);
 tmp = v10;
 v10 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v20));
 v20 = _mm256_sub_pd(_mm256_mul_pd(gamma, v20), _mm256_mul_pd(sigma, tmp));
 tmp = v11;
 v11 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v21));
 v21 = _mm256_sub_pd(_mm256_mul_pd(gamma, v21), _mm256_mul_pd(sigma, tmp));
    gamma = _mm256_broadcast_sd(&G[2 * (k + 1) ]);
    sigma = _mm256_broadcast_sd(&G[2 * (k + 1)  + 1]);
 tmp = v00;
 v00 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v10));
 v10 = _mm256_sub_pd(_mm256_mul_pd(gamma, v10), _mm256_mul_pd(sigma, tmp));
 tmp = v01;
 v01 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v11));
 v11 = _mm256_sub_pd(_mm256_mul_pd(gamma, v11), _mm256_mul_pd(sigma, tmp));
for (int g = 2; g < n - 1; g++)
{
v30= _mm256_loadu_pd(&V[i + (g + 1) * ldv ]);
v31= _mm256_loadu_pd(&V[i + (g + 1) * ldv + 4 ]);
gamma = _mm256_broadcast_sd(&G[2 * k + g * ldg]);
sigma = _mm256_broadcast_sd(&G[2 * k + g * ldg + 1]);
tmp = v20;
 v20 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v30));
 v30 = _mm256_sub_pd(_mm256_mul_pd(gamma, v30), _mm256_mul_pd(sigma, tmp));
tmp = v21;
 v21 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v31));
 v31 = _mm256_sub_pd(_mm256_mul_pd(gamma, v31), _mm256_mul_pd(sigma, tmp));
gamma = _mm256_broadcast_sd(&G[2 * (k + 1) + (g - 1) * ldg]);
sigma = _mm256_broadcast_sd(&G[2 * (k + 1) + (g - 1) * ldg + 1]);
tmp = v10;
 v10 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v20));
 v20 = _mm256_sub_pd(_mm256_mul_pd(gamma, v20), _mm256_mul_pd(sigma, tmp));
tmp = v11;
 v11 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v21));
 v21 = _mm256_sub_pd(_mm256_mul_pd(gamma, v21), _mm256_mul_pd(sigma, tmp));
gamma = _mm256_broadcast_sd(&G[2 * (k + 2) + (g - 2) * ldg]);
sigma = _mm256_broadcast_sd(&G[2 * (k + 2) + (g - 2) * ldg + 1]);
tmp = v00;
 v00 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v10));
 v10 = _mm256_sub_pd(_mm256_mul_pd(gamma, v10), _mm256_mul_pd(sigma, tmp));
tmp = v01;
 v01 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v11));
 v11 = _mm256_sub_pd(_mm256_mul_pd(gamma, v11), _mm256_mul_pd(sigma, tmp));
_mm256_storeu_pd(&V[i + (g-2) * ldv ], v00);
_mm256_storeu_pd(&V[i + (g-2) * ldv + 4], v01);
v00=v10;
v01=v11;
v10=v20;
v11=v21;
v20=v30;
v21=v31;
}
gamma = _mm256_broadcast_sd(&G[2 * (k + 1) + (n - 2) * ldg]);
sigma = _mm256_broadcast_sd(&G[2 * (k + 1) + (n - 2) * ldg + 1]);
tmp = v10;
 v10 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v20));
 v20 = _mm256_sub_pd(_mm256_mul_pd(gamma, v20), _mm256_mul_pd(sigma, tmp));
tmp = v11;
 v11 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v21));
 v21 = _mm256_sub_pd(_mm256_mul_pd(gamma, v21), _mm256_mul_pd(sigma, tmp));
gamma = _mm256_broadcast_sd(&G[2 * (k + 2) + (n - 3) * ldg]);
sigma = _mm256_broadcast_sd(&G[2 * (k + 2) + (n - 3) * ldg + 1]);
tmp = v00;
 v00 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v10));
 v10 = _mm256_sub_pd(_mm256_mul_pd(gamma, v10), _mm256_mul_pd(sigma, tmp));
tmp = v01;
 v01 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v11));
 v11 = _mm256_sub_pd(_mm256_mul_pd(gamma, v11), _mm256_mul_pd(sigma, tmp));
gamma = _mm256_broadcast_sd(&G[2 * (k + 2) + (n - 2) * ldg]);
sigma = _mm256_broadcast_sd(&G[2 * (k + 2) + (n - 2) * ldg + 1]);
tmp = v10;
 v10 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v20));
 v20 = _mm256_sub_pd(_mm256_mul_pd(gamma, v20), _mm256_mul_pd(sigma, tmp));
tmp = v11;
 v11 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v21));
 v21 = _mm256_sub_pd(_mm256_mul_pd(gamma, v21), _mm256_mul_pd(sigma, tmp));
_mm256_storeu_pd(&V[i + (n - 3) * ldv], v00);
_mm256_storeu_pd(&V[i + (n - 3) * ldv + 4], v01);
_mm256_storeu_pd(&V[i + (n - 2) * ldv], v10);
_mm256_storeu_pd(&V[i + (n - 2) * ldv + 4], v11);
_mm256_storeu_pd(&V[i + (n - 1) * ldv], v20);
_mm256_storeu_pd(&V[i + (n - 1) * ldv + 4], v21);
}
