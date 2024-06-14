#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <pmmintrin.h>
#include "apply_rev_avx.h" 
void apply_rev_avx_mv(int k, int m, int n, double *G, double *V, int ldv, int ldg, int i)
{
__m256d  v00,  v10,  v20,  v30, gamma, sigma, tmp;
v00 = _mm256_loadu_pd(&V[i]);
v10 = _mm256_loadu_pd(&V[i + ldv]);
v20 = _mm256_loadu_pd(&V[i + 2 * ldv]);
    gamma = _mm256_broadcast_sd(&G[2 * 0 + k * ldg]);
    sigma = _mm256_broadcast_sd(&G[2 * 0 + k * ldg + 1]);
 tmp = v00;
 v00 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v10));
 v10 = _mm256_sub_pd(_mm256_mul_pd(gamma, v10), _mm256_mul_pd(sigma, tmp));
    gamma = _mm256_broadcast_sd(&G[2 * 1  + k * ldg]);
    sigma = _mm256_broadcast_sd(&G[2 * 1  + k * ldg + 1]);
 tmp = v10;
 v10 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v20));
 v20 = _mm256_sub_pd(_mm256_mul_pd(gamma, v20), _mm256_mul_pd(sigma, tmp));
    gamma = _mm256_broadcast_sd(&G[2 * 0 + (k + 1) * ldg ]);
    sigma = _mm256_broadcast_sd(&G[2 * 0 + (k + 1) * ldg  + 1]);
 tmp = v00;
 v00 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v10));
 v10 = _mm256_sub_pd(_mm256_mul_pd(gamma, v10), _mm256_mul_pd(sigma, tmp));
for (int g = 2; g < n - 1; g++)
{
v30= _mm256_loadu_pd(&V[i + (g + 1) * ldv ]);
gamma = _mm256_broadcast_sd(&G[2 * g + k * ldg]);
sigma = _mm256_broadcast_sd(&G[2 * g + k * ldg + 1]);
tmp = v20;
 v20 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v30));
 v30 = _mm256_sub_pd(_mm256_mul_pd(gamma, v30), _mm256_mul_pd(sigma, tmp));
gamma = _mm256_broadcast_sd(&G[2 * (g - 1) + (k + 1) * ldg]);
sigma = _mm256_broadcast_sd(&G[2 * (g - 1) + (k + 1) * ldg + 1]);
tmp = v10;
 v10 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v20));
 v20 = _mm256_sub_pd(_mm256_mul_pd(gamma, v20), _mm256_mul_pd(sigma, tmp));
gamma = _mm256_broadcast_sd(&G[2 * (g - 2) + (k + 2) * ldg]);
sigma = _mm256_broadcast_sd(&G[2 * (g - 2) + (k + 2) * ldg + 1]);
tmp = v00;
 v00 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v10));
 v10 = _mm256_sub_pd(_mm256_mul_pd(gamma, v10), _mm256_mul_pd(sigma, tmp));
_mm256_storeu_pd(&V[i + (g-2) * ldv ], v00);
v00=v10;
v10=v20;
v20=v30;
}
gamma = _mm256_broadcast_sd(&G[2 * (n - 2) + (k + 1) * ldg]);
sigma = _mm256_broadcast_sd(&G[2 * (n - 2) + (k + 1) * ldg + 1]);
tmp = v10;
 v10 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v20));
 v20 = _mm256_sub_pd(_mm256_mul_pd(gamma, v20), _mm256_mul_pd(sigma, tmp));
gamma = _mm256_broadcast_sd(&G[2 * (n - 3) + (k + 2) * ldg]);
sigma = _mm256_broadcast_sd(&G[2 * (n - 3) + (k + 2) * ldg + 1]);
tmp = v00;
 v00 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v10));
 v10 = _mm256_sub_pd(_mm256_mul_pd(gamma, v10), _mm256_mul_pd(sigma, tmp));
gamma = _mm256_broadcast_sd(&G[2 * (n - 2) + (k + 2) * ldg]);
sigma = _mm256_broadcast_sd(&G[2 * (n - 2) + (k + 2) * ldg + 1]);
tmp = v10;
 v10 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v20));
 v20 = _mm256_sub_pd(_mm256_mul_pd(gamma, v20), _mm256_mul_pd(sigma, tmp));
_mm256_storeu_pd(&V[i + (n - 3) * ldv], v00);
_mm256_storeu_pd(&V[i + (n - 2) * ldv], v10);
_mm256_storeu_pd(&V[i + (n - 1) * ldv], v20);
}
void apply_rev_avx_mv_seq(int k,int m, int n, double *G, double *V,int ldg)
{
__m256d  v00,  v10,  v20,  v30, gamma, sigma, tmp;
v00 = _mm256_loadu_pd(&V[0 + 0]);
v10 = _mm256_loadu_pd(&V[4 + 0]);
v20 = _mm256_loadu_pd(&V[8 + 0]);
    gamma = _mm256_broadcast_sd(&G[2 * 0 + k * ldg]);
    sigma = _mm256_broadcast_sd(&G[2 * 0 + k * ldg + 1]);
 tmp = v00;
 v00 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v10));
 v10 = _mm256_sub_pd(_mm256_mul_pd(gamma, v10), _mm256_mul_pd(sigma, tmp));
    gamma = _mm256_broadcast_sd(&G[2 * 1  + k * ldg]);
    sigma = _mm256_broadcast_sd(&G[2 * 1  + k * ldg + 1]);
 tmp = v10;
 v10 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v20));
 v20 = _mm256_sub_pd(_mm256_mul_pd(gamma, v20), _mm256_mul_pd(sigma, tmp));
    gamma = _mm256_broadcast_sd(&G[2 * 0 + (k + 1) * ldg ]);
    sigma = _mm256_broadcast_sd(&G[2 * 0 + (k + 1) * ldg  + 1]);
 tmp = v00;
 v00 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v10));
 v10 = _mm256_sub_pd(_mm256_mul_pd(gamma, v10), _mm256_mul_pd(sigma, tmp));
for (int g = 2; g < n - 1; g++)
{
v30 = _mm256_loadu_pd(&V[(g + 1) * 4 + 0]);
gamma = _mm256_broadcast_sd(&G[2 * g + k * ldg]);
sigma = _mm256_broadcast_sd(&G[2 * g + k * ldg + 1]);
tmp = v20;
 v20 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v30));
 v30 = _mm256_sub_pd(_mm256_mul_pd(gamma, v30), _mm256_mul_pd(sigma, tmp));
gamma = _mm256_broadcast_sd(&G[2 * (g - 1) + (k + 1) * ldg]);
sigma = _mm256_broadcast_sd(&G[2 * (g - 1) + (k + 1) * ldg + 1]);
tmp = v10;
 v10 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v20));
 v20 = _mm256_sub_pd(_mm256_mul_pd(gamma, v20), _mm256_mul_pd(sigma, tmp));
gamma = _mm256_broadcast_sd(&G[2 * (g - 2) + (k + 2) * ldg]);
sigma = _mm256_broadcast_sd(&G[2 * (g - 2) + (k + 2) * ldg + 1]);
tmp = v00;
 v00 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v10));
 v10 = _mm256_sub_pd(_mm256_mul_pd(gamma, v10), _mm256_mul_pd(sigma, tmp));
_mm256_storeu_pd(&V[4 * (g - 2)], v00);
v00=v10;
v10=v20;
v20=v30;
}
gamma = _mm256_broadcast_sd(&G[2 * (n - 2) + (k + 1) * ldg]);
sigma = _mm256_broadcast_sd(&G[2 * (n - 2) + (k + 1) * ldg + 1]);
tmp = v10;
 v10 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v20));
 v20 = _mm256_sub_pd(_mm256_mul_pd(gamma, v20), _mm256_mul_pd(sigma, tmp));
gamma = _mm256_broadcast_sd(&G[2 * (n - 3) + (k + 2) * ldg]);
sigma = _mm256_broadcast_sd(&G[2 * (n - 3) + (k + 2) * ldg + 1]);
tmp = v00;
 v00 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v10));
 v10 = _mm256_sub_pd(_mm256_mul_pd(gamma, v10), _mm256_mul_pd(sigma, tmp));
gamma = _mm256_broadcast_sd(&G[2 * (n - 2) + (k + 2) * ldg]);
sigma = _mm256_broadcast_sd(&G[2 * (n - 2) + (k + 2) * ldg + 1]);
tmp = v10;
 v10 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v20));
 v20 = _mm256_sub_pd(_mm256_mul_pd(gamma, v20), _mm256_mul_pd(sigma, tmp));
_mm256_storeu_pd(&V[4 * (n - 3)], v00);
_mm256_storeu_pd(&V[4 * (n - 2)], v10);
_mm256_storeu_pd(&V[4 * (n - 1)], v20);
}
