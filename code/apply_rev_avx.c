#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <pmmintrin.h>
#include "apply_rev_avx.h"
void apply_rev_avx_mv(int k, int m, int n, double *G, double *V, int ldv, int ldg, int my, int i, int mv)
{
__m256d v0,v1,v2,v3,v4,v5,v6,v7,gamma, sigma, tmp;
    v0 = _mm256_loadu_pd(&V[i]);
    v1 = _mm256_loadu_pd(&V[i + 4]);
    v2 = _mm256_loadu_pd(&V[i + ldv]);
    v3 = _mm256_loadu_pd(&V[i + ldv + 4]);
    v4 = _mm256_loadu_pd(&V[i+ 2 * ldv ]);
    v5 = _mm256_loadu_pd(&V[i+ 2 * ldv  + 4]);
    gamma = _mm256_broadcast_sd(&G[2 * k ]);
    sigma = _mm256_broadcast_sd(&G[2 * k  + 1]);
    tmp = v0;
    v0 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v2));
    v2 = _mm256_sub_pd(_mm256_mul_pd(gamma, v2), _mm256_mul_pd(sigma, tmp));
    tmp = v1;
    v1 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v3));
    v3 = _mm256_sub_pd(_mm256_mul_pd(gamma, v3), _mm256_mul_pd(sigma, tmp));
    for (int g = 1; g < n - 1; g++)
    {
        v6 = _mm256_loadu_pd(&V[i + (g + 1) * ldv]);
        v7 = _mm256_loadu_pd(&V[i + (g + 1) * ldv + 4]);
        gamma = _mm256_broadcast_sd(&G[2 * k  + g * ldg]);
        sigma = _mm256_broadcast_sd(&G[2 * k  + g * ldg + 1]);
        tmp = v2;
        v2 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v4));
        v4 = _mm256_sub_pd(_mm256_mul_pd(gamma, v4), _mm256_mul_pd(sigma, tmp));
        tmp = v3;
        v3 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v5));
        v5 = _mm256_sub_pd(_mm256_mul_pd(gamma, v5), _mm256_mul_pd(sigma, tmp));
        gamma = _mm256_broadcast_sd(&G[2 * (k + 1)  + (g - 1) * ldg ]);
        sigma = _mm256_broadcast_sd(&G[2 * (k + 1)  + (g - 1) * ldg  + 1]);
        tmp = v0;
        v0 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v2));
        v2 = _mm256_sub_pd(_mm256_mul_pd(gamma, v2), _mm256_mul_pd(sigma, tmp));
        tmp = v1;
        v1 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v3));
        v3 = _mm256_sub_pd(_mm256_mul_pd(gamma, v3), _mm256_mul_pd(sigma, tmp));
        gamma = _mm256_broadcast_sd(&G[2 * (k + 2)  + (g - 2) * ldg ]);
        sigma = _mm256_broadcast_sd(&G[2 * (k + 2)  + (g - 2) * ldg  + 1]);
        tmp = v-2;
        v-2 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v0));
        v0 = _mm256_sub_pd(_mm256_mul_pd(gamma, v0), _mm256_mul_pd(sigma, tmp));
        tmp = v-1;
        v-1 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v1));
        v1 = _mm256_sub_pd(_mm256_mul_pd(gamma, v1), _mm256_mul_pd(sigma, tmp));
        _mm256_storeu_pd(&V[i + (g - 2) * ldv], v0);
        _mm256_storeu_pd(&V[i + (g - 2) * ldv + 4], v1);
        v0 = v2;
        v1 = v3;
        v2 = v4;
        v3 = v5;
        v4 = v6;
        v5 = v7;
    }
    gamma = _mm256_broadcast_sd(&G[2 * k + 1 * ldg ]);
    sigma = _mm256_broadcast_sd(&G[2 * k + 1 * ldg  + 1]);
    tmp = v0;
    v0 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v2));
    v2 = _mm256_sub_pd(_mm256_mul_pd(gamma, v2), _mm256_mul_pd(sigma, tmp));
    tmp = v1;
    v1 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v3));
    v3 = _mm256_sub_pd(_mm256_mul_pd(gamma, v3), _mm256_mul_pd(sigma, tmp));
    for (int g = 1; g < n - 1; g++)
    {
        v6 = _mm256_loadu_pd(&V[i + (g + 1) * ldv]);
        v7 = _mm256_loadu_pd(&V[i + (g + 1) * ldv + 4]);
        gamma = _mm256_broadcast_sd(&G[2 * k  + g * ldg]);
        sigma = _mm256_broadcast_sd(&G[2 * k  + g * ldg + 1]);
        tmp = v2;
        v2 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v4));
        v4 = _mm256_sub_pd(_mm256_mul_pd(gamma, v4), _mm256_mul_pd(sigma, tmp));
        tmp = v3;
        v3 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v5));
        v5 = _mm256_sub_pd(_mm256_mul_pd(gamma, v5), _mm256_mul_pd(sigma, tmp));
        gamma = _mm256_broadcast_sd(&G[2 * (k + 1)  + (g - 1) * ldg ]);
        sigma = _mm256_broadcast_sd(&G[2 * (k + 1)  + (g - 1) * ldg  + 1]);
        tmp = v0;
        v0 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v2));
        v2 = _mm256_sub_pd(_mm256_mul_pd(gamma, v2), _mm256_mul_pd(sigma, tmp));
        tmp = v1;
        v1 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v3));
        v3 = _mm256_sub_pd(_mm256_mul_pd(gamma, v3), _mm256_mul_pd(sigma, tmp));
        gamma = _mm256_broadcast_sd(&G[2 * (k + 2)  + (g - 2) * ldg ]);
        sigma = _mm256_broadcast_sd(&G[2 * (k + 2)  + (g - 2) * ldg  + 1]);
        tmp = v-2;
        v-2 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v0));
        v0 = _mm256_sub_pd(_mm256_mul_pd(gamma, v0), _mm256_mul_pd(sigma, tmp));
        tmp = v-1;
        v-1 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v1));
        v1 = _mm256_sub_pd(_mm256_mul_pd(gamma, v1), _mm256_mul_pd(sigma, tmp));
        _mm256_storeu_pd(&V[i + (g - 2) * ldv], v0);
        _mm256_storeu_pd(&V[i + (g - 2) * ldv + 4], v1);
        v0 = v2;
        v1 = v3;
        v2 = v4;
        v3 = v5;
        v4 = v6;
        v5 = v7;
    }
    gamma = _mm256_broadcast_sd(&G[2 * (k + 1) ]);
    sigma = _mm256_broadcast_sd(&G[2 * (k + 1)  + 1]);
    tmp = v0;
    v0 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v2));
    v2 = _mm256_sub_pd(_mm256_mul_pd(gamma, v2), _mm256_mul_pd(sigma, tmp));
    tmp = v1;
    v1 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v3));
    v3 = _mm256_sub_pd(_mm256_mul_pd(gamma, v3), _mm256_mul_pd(sigma, tmp));
    for (int g = 1; g < n - 1; g++)
    {
        v6 = _mm256_loadu_pd(&V[i + (g + 1) * ldv]);
        v7 = _mm256_loadu_pd(&V[i + (g + 1) * ldv + 4]);
        gamma = _mm256_broadcast_sd(&G[2 * k  + g * ldg]);
        sigma = _mm256_broadcast_sd(&G[2 * k  + g * ldg + 1]);
        tmp = v2;
        v2 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v4));
        v4 = _mm256_sub_pd(_mm256_mul_pd(gamma, v4), _mm256_mul_pd(sigma, tmp));
        tmp = v3;
        v3 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v5));
        v5 = _mm256_sub_pd(_mm256_mul_pd(gamma, v5), _mm256_mul_pd(sigma, tmp));
        gamma = _mm256_broadcast_sd(&G[2 * (k + 1)  + (g - 1) * ldg ]);
        sigma = _mm256_broadcast_sd(&G[2 * (k + 1)  + (g - 1) * ldg  + 1]);
        tmp = v0;
        v0 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v2));
        v2 = _mm256_sub_pd(_mm256_mul_pd(gamma, v2), _mm256_mul_pd(sigma, tmp));
        tmp = v1;
        v1 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v3));
        v3 = _mm256_sub_pd(_mm256_mul_pd(gamma, v3), _mm256_mul_pd(sigma, tmp));
        gamma = _mm256_broadcast_sd(&G[2 * (k + 2)  + (g - 2) * ldg ]);
        sigma = _mm256_broadcast_sd(&G[2 * (k + 2)  + (g - 2) * ldg  + 1]);
        tmp = v-2;
        v-2 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v0));
        v0 = _mm256_sub_pd(_mm256_mul_pd(gamma, v0), _mm256_mul_pd(sigma, tmp));
        tmp = v-1;
        v-1 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v1));
        v1 = _mm256_sub_pd(_mm256_mul_pd(gamma, v1), _mm256_mul_pd(sigma, tmp));
        _mm256_storeu_pd(&V[i + (g - 2) * ldv], v0);
        _mm256_storeu_pd(&V[i + (g - 2) * ldv + 4], v1);
        v0 = v2;
        v1 = v3;
        v2 = v4;
        v3 = v5;
        v4 = v6;
        v5 = v7;
    }
    gamma = _mm256_broadcast_sd(&G[2 * (k + 1) + (n - 3) * ldg]);
    sigma = _mm256_broadcast_sd(&G[2 * (k + 1) + (n - 3) * ldg + 1]);
    tmp = v0;
    v0 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v2));
    v2 = _mm256_sub_pd(_mm256_mul_pd(gamma, v2), _mm256_mul_pd(sigma, tmp));
    tmp = v1;
    v1 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v3));
    v3 = _mm256_sub_pd(_mm256_mul_pd(gamma, v3), _mm256_mul_pd(sigma, tmp));
    gamma = _mm256_broadcast_sd(&G[2 * (k + 2) + (n - 4) * ldg]);
    sigma = _mm256_broadcast_sd(&G[2 * (k + 2) + (n - 4) * ldg + 1]);
    tmp = v0;
    v0 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v2));
    v2 = _mm256_sub_pd(_mm256_mul_pd(gamma, v2), _mm256_mul_pd(sigma, tmp));
    tmp = v1;
    v1 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v3));
    v3 = _mm256_sub_pd(_mm256_mul_pd(gamma, v3), _mm256_mul_pd(sigma, tmp));
    gamma = _mm256_broadcast_sd(&G[2 * (k + 2) + (n - 2) * ldg]);
    sigma = _mm256_broadcast_sd(&G[2 * (k + 2) + (n - 2) * ldg + 1]);
    tmp = v0;
    v0 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v2));
    v2 = _mm256_sub_pd(_mm256_mul_pd(gamma, v2), _mm256_mul_pd(sigma, tmp));
    tmp = v1;
    v1 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v3));
    v3 = _mm256_sub_pd(_mm256_mul_pd(gamma, v3), _mm256_mul_pd(sigma, tmp));
    _mm256_storeu_pd(&V[i + (n - 3) * ldv], v0);
    _mm256_storeu_pd(&V[i + (n - 3) * ldv + 4], v1);
    _mm256_storeu_pd(&V[i + (n - 2) * ldv], v2);
    _mm256_storeu_pd(&V[i + (n - 2) * ldv + 4], v3);
    _mm256_storeu_pd(&V[i + (n - 1) * ldv], v4);
    _mm256_storeu_pd(&V[i + (n - 1) * ldv + 4], v5);
}
