#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <pmmintrin.h>
#include "apply_rev_avx.h"
void apply_rev_avx512_mv(int k, int m, int n, double *G, double *V, int ldv, int ldg, int i)
{
    __m512d v00, v10, v20, gamma, sigma, tmp;
    v00 = _mm512_loadu_pd(&V[i]);
    v10 = _mm512_loadu_pd(&V[i + ldv]);
    gamma = _mm512_set1_pd(G[2 * k]);
    sigma = _mm512_set1_pd(G[2 * k + 1]);
    tmp = v00;
    v00 = _mm512_add_pd(_mm512_mul_pd(gamma, tmp), _mm512_mul_pd(sigma, v10));
    v10 = _mm512_sub_pd(_mm512_mul_pd(gamma, v10), _mm512_mul_pd(sigma, tmp));
    for (int g = 1; g < n - 1; g++)
    {
        v20 = _mm512_loadu_pd(&V[i + (g + 1) * ldv]);
        gamma = _mm512_set1_pd(G[2 * k + g * ldg]);
        sigma = _mm512_set1_pd(G[2 * k + g * ldg + 1]);
        tmp = v10;
        v10 = _mm512_add_pd(_mm512_mul_pd(gamma, tmp), _mm512_mul_pd(sigma, v20));
        v20 = _mm512_sub_pd(_mm512_mul_pd(gamma, v20), _mm512_mul_pd(sigma, tmp));
        gamma = _mm512_set1_pd(G[2 * (k + 1) + (g - 1) * ldg]);
        sigma = _mm512_set1_pd(G[2 * (k + 1) + (g - 1) * ldg + 1]);
        tmp = v00;
        v00 = _mm512_add_pd(_mm512_mul_pd(gamma, tmp), _mm512_mul_pd(sigma, v10));
        v10 = _mm512_sub_pd(_mm512_mul_pd(gamma, v10), _mm512_mul_pd(sigma, tmp));
        _mm512_storeu_pd(&V[i + (g - 1) * ldv], v00);
        v00 = v10;
        v10 = v20;
    }
    gamma = _mm512_set1_pd(G[2 * (k + 1) + (n - 2) * ldg]);
    sigma = _mm512_set1_pd(G[2 * (k + 1) + (n - 2) * ldg + 1]);
    tmp = v00;
    v00 = _mm512_add_pd(_mm512_mul_pd(gamma, tmp), _mm512_mul_pd(sigma, v10));
    v10 = _mm512_sub_pd(_mm512_mul_pd(gamma, v10), _mm512_mul_pd(sigma, tmp));
    _mm512_storeu_pd(&V[i + (n - 2) * ldv], v00);
    _mm512_storeu_pd(&V[i + (n - 1) * ldv], v10);
}
void apply_rev_avx_mv(int k, int m, int n, double *G, double *V, int ldv, int ldg, int i)
{
    __m256d v00, v10, v20, gamma, sigma, tmp;
    v00 = _mm256_loadu_pd(&V[i]);
    v10 = _mm256_loadu_pd(&V[i + ldv]);
    gamma = _mm256_broadcast_sd(&G[2 * k]);
    sigma = _mm256_broadcast_sd(&G[2 * k + 1]);
    tmp = v00;
    v00 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v10));
    v10 = _mm256_sub_pd(_mm256_mul_pd(gamma, v10), _mm256_mul_pd(sigma, tmp));
    for (int g = 1; g < n - 1; g++)
    {
        v20 = _mm256_loadu_pd(&V[i + (g + 1) * ldv]);
        gamma = _mm256_broadcast_sd(&G[2 * k + g * ldg]);
        sigma = _mm256_broadcast_sd(&G[2 * k + g * ldg + 1]);
        tmp = v10;
        v10 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v20));
        v20 = _mm256_sub_pd(_mm256_mul_pd(gamma, v20), _mm256_mul_pd(sigma, tmp));
        gamma = _mm256_broadcast_sd(&G[2 * (k + 1) + (g - 1) * ldg]);
        sigma = _mm256_broadcast_sd(&G[2 * (k + 1) + (g - 1) * ldg + 1]);
        tmp = v00;
        v00 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v10));
        v10 = _mm256_sub_pd(_mm256_mul_pd(gamma, v10), _mm256_mul_pd(sigma, tmp));
        _mm256_storeu_pd(&V[i + (g - 1) * ldv], v00);
        v00 = v10;
        v10 = v20;
    }
    gamma = _mm256_broadcast_sd(&G[2 * (k + 1) + (n - 2) * ldg]);
    sigma = _mm256_broadcast_sd(&G[2 * (k + 1) + (n - 2) * ldg + 1]);
    tmp = v00;
    v00 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v10));
    v10 = _mm256_sub_pd(_mm256_mul_pd(gamma, v10), _mm256_mul_pd(sigma, tmp));
    _mm256_storeu_pd(&V[i + (n - 2) * ldv], v00);
    _mm256_storeu_pd(&V[i + (n - 1) * ldv], v10);
}
