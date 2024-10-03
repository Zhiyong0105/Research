#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <pmmintrin.h>
#include "apply_rev_avx.h" 
void apply_rev_avx_mv(int k, int m, int n, double *G, double *V, int ldv, int ldg, int i)
{
__m256d  v00,  v01,  v02,  v10,  v11,  v12, gamma, sigma, tmp;
v00 = _mm256_loadu_pd(&V[i]);
v01 = _mm256_loadu_pd(&V[i + 4 * 1]);
v02 = _mm256_loadu_pd(&V[i + 4 * 2]);
for (int g = 0; g < n - 1; g++)
{
v10= _mm256_loadu_pd(&V[i + (g + 1) * ldv ]);
v11= _mm256_loadu_pd(&V[i + (g + 1) * ldv + 4 * 1 ]);
v12= _mm256_loadu_pd(&V[i + (g + 1) * ldv + 4 * 2 ]);
gamma = _mm256_broadcast_sd(&G[2 * g + k * ldg]);
sigma = _mm256_broadcast_sd(&G[2 * g + k * ldg + 1]);
tmp = v00;
 v00 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v10));
 v10 = _mm256_sub_pd(_mm256_mul_pd(gamma, v10), _mm256_mul_pd(sigma, tmp));
tmp = v01;
 v01 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v11));
 v11 = _mm256_sub_pd(_mm256_mul_pd(gamma, v11), _mm256_mul_pd(sigma, tmp));
tmp = v02;
 v02 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v12));
 v12 = _mm256_sub_pd(_mm256_mul_pd(gamma, v12), _mm256_mul_pd(sigma, tmp));
_mm256_storeu_pd(&V[i + (g-0) * ldv ], v00);
_mm256_storeu_pd(&V[i + (g-0) * ldv + 4 * 1], v01);
_mm256_storeu_pd(&V[i + (g-0) * ldv + 4 * 2], v02);
v00=v10;
v01=v11;
v02=v12;
}
_mm256_storeu_pd(&V[i + (n - 1) * ldv], v00);
_mm256_storeu_pd(&V[i + (n - 1) * ldv + 4 * 1], v01);
_mm256_storeu_pd(&V[i + (n - 1) * ldv + 4 * 2], v02);
}
void apply_rev_avx_mv_seq(int k,int m, int n, double *G, double *V,int ldg)
{
__m512d  v00,  v01,  v02,  v10,  v11,  v12, gamma, sigma, tmp;
v00 = _mm512_loadu_pd(&V[0 + 0]);
v01 = _mm512_loadu_pd(&V[0 + 8]);
v02 = _mm512_loadu_pd(&V[0 + 16]);
for (int g = 0; g < n - 1; g++)
{
v10 = _mm512_loadu_pd(&V[(g + 1) * 24 + 0]);
v11 = _mm512_loadu_pd(&V[(g + 1) * 24 + 8]);
v12 = _mm512_loadu_pd(&V[(g + 1) * 24 + 16]);
gamma = _mm512_set1_pd(G[2 * g + k * ldg]);
sigma = _mm512_set1_pd(G[2 * g + k * ldg + 1]);
tmp = v00;
 v00 = _mm512_add_pd(_mm512_mul_pd(gamma, tmp), _mm512_mul_pd(sigma, v10));
 v10 = _mm512_sub_pd(_mm512_mul_pd(gamma, v10), _mm512_mul_pd(sigma, tmp));
tmp = v01;
 v01 = _mm512_add_pd(_mm512_mul_pd(gamma, tmp), _mm512_mul_pd(sigma, v11));
 v11 = _mm512_sub_pd(_mm512_mul_pd(gamma, v11), _mm512_mul_pd(sigma, tmp));
tmp = v02;
 v02 = _mm512_add_pd(_mm512_mul_pd(gamma, tmp), _mm512_mul_pd(sigma, v12));
 v12 = _mm512_sub_pd(_mm512_mul_pd(gamma, v12), _mm512_mul_pd(sigma, tmp));
_mm512_storeu_pd(&V[24 * (g - 0)], v00);
_mm512_storeu_pd(&V[24 * (g - 0) + 8 * 1], v01);
_mm512_storeu_pd(&V[24 * (g - 0) + 8 * 2], v02);
v00=v10;
v01=v11;
v02=v12;
}
_mm512_storeu_pd(&V[24 * (n - 1)], v00);
_mm512_storeu_pd(&V[24 * (n - 1)+ 8 * 1], v01);
_mm512_storeu_pd(&V[24 * (n - 1)+ 8 * 2], v02);
}
void apply_rev_avx512_mv_seq(int k,int m, int n, double *G, double *V,int ldg)
{
__m512d  v00,  v01,  v02,  v10,  v11,  v12, gamma, sigma, tmp;
v00 = _mm512_loadu_pd(&V[0 + 0]);
v01 = _mm512_loadu_pd(&V[0 + 8]);
v02 = _mm512_loadu_pd(&V[0 + 16]);
for (int g = 0; g < n - 1; g++)
{
v10 = _mm512_loadu_pd(&V[(g + 1) * 24 + 0]);
v11 = _mm512_loadu_pd(&V[(g + 1) * 24 + 4]);
v12 = _mm512_loadu_pd(&V[(g + 1) * 24 + 8]);
gamma = _mm512_set1_pd(G[2 * g + k * ldg]);
sigma = _mm512_set1_pd(G[2 * g + k * ldg + 1]);
tmp = v00;
 v00 = _mm512_add_pd(_mm512_mul_pd(gamma, tmp), _mm512_mul_pd(sigma, v10));
 v10 = _mm512_sub_pd(_mm512_mul_pd(gamma, v10), _mm512_mul_pd(sigma, tmp));
tmp = v01;
 v01 = _mm512_add_pd(_mm512_mul_pd(gamma, tmp), _mm512_mul_pd(sigma, v11));
 v11 = _mm512_sub_pd(_mm512_mul_pd(gamma, v11), _mm512_mul_pd(sigma, tmp));
tmp = v02;
 v02 = _mm512_add_pd(_mm512_mul_pd(gamma, tmp), _mm512_mul_pd(sigma, v12));
 v12 = _mm512_sub_pd(_mm512_mul_pd(gamma, v12), _mm512_mul_pd(sigma, tmp));
_mm512_storeu_pd(&V[24 * (g - 0)], v00);
_mm512_storeu_pd(&V[24 * (g - 0) + 8 * 1], v01);
_mm512_storeu_pd(&V[24 * (g - 0) + 8 * 2], v02);
v00=v10;
v01=v11;
v02=v12;
}
_mm512_storeu_pd(&V[24 * (n - 1)], v00);
_mm512_storeu_pd(&V[24 * (n - 1)+ 8 * 1], v01);
_mm512_storeu_pd(&V[24 * (n - 1)+ 8 * 2], v02);
}
