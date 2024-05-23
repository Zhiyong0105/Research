#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <pmmintrin.h>
#include "apply_rev_avx.h"
void apply_rev_avx(int k, int m, int n, double *G, double *V, int ldv, int ldg, int my, int i)
{
    __m256d v0, v1, v2, gamma, sigma, tmp;
    v0 = _mm256_loadu_pd(&V[i + 0 * ldv]);
    v1 = _mm256_loadu_pd(&V[i + 1 * ldv]);
    gamma = _mm256_broadcast_sd(&G[2 * (k + 0) + 0 * ldg]);
    sigma = _mm256_broadcast_sd(&G[2 * (k + 0) + 0 * ldg + 1]);
    tmp = v0;
    v0 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v1));
    v1 = _mm256_sub_pd(_mm256_mul_pd(gamma, v1), _mm256_mul_pd(sigma, tmp));
    for (int g = 1; g < n - 1; g++)
    {
        v2 = _mm256_loadu_pd(&V[i + (g + 1) * ldv]);
        gamma = _mm256_broadcast_sd(&G[2 * (k + 0) + (g - 0) * ldg]);
        sigma = _mm256_broadcast_sd(&G[2 * (k + 0) + (g - 0) * ldg + 1]);
        tmp = v1;
        v1 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v2));
        v2 = _mm256_sub_pd(_mm256_mul_pd(gamma, v2), _mm256_mul_pd(sigma, tmp));
        gamma = _mm256_broadcast_sd(&G[2 * (k + 1) + (g - 1) * ldg]);
        sigma = _mm256_broadcast_sd(&G[2 * (k + 1) + (g - 1) * ldg + 1]);
        tmp = v0;
        v0 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v1));
        v1 = _mm256_sub_pd(_mm256_mul_pd(gamma, v1), _mm256_mul_pd(sigma, tmp));
        _mm256_storeu_pd(&V[i + (g - 1) * ldv], v0);
        v0 = v1;
        v1 = v2;
    }
    gamma = _mm256_broadcast_sd(&G[2 * (k + 1) + (n - 2) * ldg]);
    sigma = _mm256_broadcast_sd(&G[2 * (k + 1) + (n - 2) * ldg + 1]);
    tmp = v0;
    v0 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v1));
    v1 = _mm256_sub_pd(_mm256_mul_pd(gamma, v1), _mm256_mul_pd(sigma, tmp));
    _mm256_storeu_pd(&V[i + (n - 2) * ldv], v0);
    _mm256_storeu_pd(&V[i + (n - 1) * ldv], v1);
}
