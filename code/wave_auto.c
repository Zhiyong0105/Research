#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <stdint.h>
#include <omp.h>
#include <pmmintrin.h>
#include <immintrin.h>
#include <sys/mman.h>
#include "function.h"
#define EPSILON 1e-12
int64_t
i64time()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * (1000000000ull) + (int64_t)ts.tv_nsec;
}
void *hugealloc(size_t size)
{
    size_t pagesize = 2 * 1024 * 1024;
    size = (size + pagesize - 1) / pagesize * pagesize;
    void *base_ptr_ = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    madvise(base_ptr_, size, MADV_HUGEPAGE);
    return base_ptr_;
}
void hugefree(void *ptr, size_t size)
{
    size_t pagesize = 2 * 1024 * 1024;
    size = (size + pagesize - 1) / pagesize * pagesize;
    munmap(ptr, size);
}
void freedmatrix(void *ptr, int m, int n, int lda)
{
    hugefree(ptr, sizeof(double) * lda * n);
}
double flush_cache(double t)
{
    int n = 20000000;
    double *a = (double *)malloc(sizeof(double) * n);
#pragma omp parallel for
    for (int i = 0; i < n; ++i)
        a[i] = t * i;
    for (int t = 0; t < 5; ++t)
#pragma omp parallel for
        for (int i = 0; i < n; ++i)
            a[i] = 2 * a[i] + 1;
    double x = 0.;
#pragma omp parallel for reduction(+ : x)
    for (int i = 0; i < n; ++i)
        x += a[i];
    free(a);
    return x;
}
double *dmatrix(int m, int n, int lda)
{
    assert(m > 0 && n > 0);
    assert(lda >= m);

    // double *ret = (double *)_mm_malloc(sizeof(double) * lda * n, 32);
    double *ret = (double *)hugealloc(sizeof(double) * lda * n);
    return ret;
}
void drandomM(int m, int n, double *a, int lda)
{
    for (int j = 0; j < n; ++j)
    {
        for (int i = 0; i < m; ++i)
        {
            // drand48 is an uniform random generator in [0., 1.)
            a[i + j * lda] = drand48();
        }
    }
}
void drandomG(int m, int n, double *a, int ldg)
{
    for (int j = 0; j < n; ++j)
    {
        for (int i = 0; i < m; ++i)
        {
            // drand48 is an uniform random generator in [0., 1.)
            double angle = (rand() % 360) * (M_PI / 180.0);
            a[2 * i + j * ldg] = cos(angle);
            a[2 * i + j * ldg + 1] = sin(angle);
        }
    }
}
void applywavemx2_avx(int m, double gamma, double sigma, double *x, double *y)
{
    double *restrict xp = x;
    double *restrict yp = y;
    int i;
    int m_iter = m / 4;
    int m_left = m % 4;
    __m256d gv, sv, xv, yv, tv;

    if (gamma == 1.0)
    {
        return;
    }

    gv = _mm256_broadcast_sd(&gamma);
    sv = _mm256_broadcast_sd(&sigma);

    for (i = 0; i < m_iter; i++, xp += 4, yp += 4)
    {
        xv = _mm256_loadu_pd(xp); // Use aligned load
        yv = _mm256_loadu_pd(yp); // Use aligned load
        tv = xv;
        xv = _mm256_add_pd(_mm256_mul_pd(gv, tv), _mm256_mul_pd(sv, yv));
        yv = _mm256_sub_pd(_mm256_mul_pd(gv, yv), _mm256_mul_pd(sv, tv));
        _mm256_storeu_pd(xp, xv); // Use aligned store
        _mm256_storeu_pd(yp, yv); // Use aligned store
    }

    for (i = 0; i < m_left; i++, xp++, yp++)
    {
        double tmp = *xp;
        *xp = gamma * tmp + sigma * (*yp);
        *yp = gamma * (*yp) - sigma * tmp;
    }
}

void applysingle_avx(int k, int m, int n, double *g, double *v, int ldv, int ldg)
{
    for (int h = 0; h < n - 1; h++)
    {
        for (int j = 0; j < k; j++)
        {
            double gamma = g[2 * j + h * ldg];
            double sigma = g[2 * j + h * ldg + 1];
            double *v = &v[j * ldg];
            double *v1 = &v[(j + 1) * ldg];
            applywavemx2_avx(m, gamma, sigma, v, v1);
        }
    }
}
void applywave_avx(int k, int m, int n, double *G, double *V, int ldv, int ldg)
{
    if (n < k || k == 1)
    {
        applysingle_avx(k, m, n, G, V, ldv, ldg);
    }

    // startup phase
    for (int j = 0; j < k - 1; j++)
    {
        for (int i = 0, g = j; i < j + 1; i++, g--)
        {
            //  printf("A:: %3d %3d\n", g, i);

            double gamma = G[2 * i + g * ldg];
            double sigma = G[2 * i + g * ldg + 1];
            double *v = &V[g * ldv];
            double *v1 = &V[(g + 1) * ldv];
            applywavemx2_avx(m, gamma, sigma, v, v1);
            // printf("%d%d\n",g,i);
        }
        // printf("\n");
    }

    // Pipeline phase
    for (int j = k - 1; j < n - 1; j++)
    {
        for (int i = 0, g = j; i < k; i++, g--)
        {
            //  printf("A:: %3d %3d\n", g, i);
            double gamma = G[2 * i + g * ldg];
            double sigma = G[2 * i + g * ldg + 1];
            double *v = &V[g * ldv];
            double *v1 = &V[(g + 1) * ldv];
            applywavemx2_avx(m, gamma, sigma, v, v1);
            //  printf("%d%d\n",g,i);
        }
        // printf("\n");
    }
    // Shutdown phase
    for (int j = n - k; j < n - 1; j++)
    {
        for (int i = j - n + k + 1, g = n - 2; i < k; i++, g--)
        {
            //  printf("A:: %3d %3d\n", g, i);
            double gamma = G[2 * i + g * ldg];
            double sigma = G[2 * i + g * ldg + 1];
            double *v = &V[g * ldv];
            double *v1 = &V[(g + 1) * ldv];
            applywavemx2_avx(m, gamma, sigma, v, v1);
            // printf("%d%d\n",g,i);
        }
        //  printf("\n");
    }
}
// void applywavemx2_avx_4x3(int m, double *V, double *G, int ldv, int ldg, int g, int i)
// {
//     /* pointers for columns*/
//     double *restrict v0 = &V[g * ldv];
//     double *restrict v1 = &V[(g + 1) * ldv];
//     double *restrict v2 = &V[(g + 2) * ldv];
//     double *restrict v3 = &V[(g + 3) * ldv];
//     double *restrict v4 = &V[(g + 4) * ldv];
//     double *restrict v5 = &V[(g - 1) * ldv];
//     double *restrict v6 = &V[(g - 2) * ldv];

//     /* Loop conditions*/

//     int m_iter = m / 8;
//     int m_left = m % 8;

//     /*Registers arguments*/
//     __m512d g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12; // gamma
//     __m512d s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12; // sigma

//     /*columns arguments*/
//     __m512d v0_vec, v1_vec, v2_vec, v3_vec, v4_vec, tmp_vec;

//     /*Loading */
//     g1 = _mm512_set1_pd(G[2 * i + g * ldg]);     // G(g,i)
//     s1 = _mm512_set1_pd(G[2 * i + g * ldg + 1]); // G(g,i)

//     g2 = _mm512_set1_pd(G[2 * i + (g + 1) * ldg]);     // G(g+1,i)
//     s2 = _mm512_set1_pd(G[2 * i + (g + 1) * ldg + 1]); // G(g+1,i)

//     g3 = _mm512_set1_pd(G[2 * i + (g + 2) * ldg]);     // G(g+2,i)
//     s3 = _mm512_set1_pd(G[2 * i + (g + 2) * ldg + 1]); // G(g+2,i)

//     g4 = _mm512_set1_pd(G[2 * i + (g + 3) * ldg]);     // G(g+3,i)
//     s4 = _mm512_set1_pd(G[2 * i + (g + 3) * ldg + 1]); // G(g+3,i)

//     g5 = _mm512_set1_pd(G[2 * (i + 1) + (g - 1) * ldg]);     // G(g-1,i+1)
//     s5 = _mm512_set1_pd(G[2 * (i + 1) + (g - 1) * ldg + 1]); // G(g-1,i+1)

//     g6 = _mm512_set1_pd(G[2 * (i + 1) + g * ldg]);     // G(g,i+1)
//     s6 = _mm512_set1_pd(G[2 * (i + 1) + g * ldg + 1]); // G(g,i+1)

//     g7 = _mm512_set1_pd(G[2 * (i + 1) + (g + 1) * ldg]);     // G(g+1,i+1)
//     s7 = _mm512_set1_pd(G[2 * (i + 1) + (g + 1) * ldg + 1]); // G(g+1,i+1)

//     g8 = _mm512_set1_pd(G[2 * (i + 1) + (g + 2) * ldg]);     // G(g+2,i+1)
//     s8 = _mm512_set1_pd(G[2 * (i + 1) + (g + 2) * ldg + 1]); // G(g+2,i+1)

//     g9 = _mm512_set1_pd(G[2 * (i + 2) + (g - 2) * ldg]);     // G(g-2,i+2)
//     s9 = _mm512_set1_pd(G[2 * (i + 2) + (g - 2) * ldg + 1]); // G(g-2,i+2)

//     g10 = _mm512_set1_pd(G[2 * (i + 2) + (g - 1) * ldg]);     // G(g-1,i+2)
//     s10 = _mm512_set1_pd(G[2 * (i + 2) + (g - 1) * ldg + 1]); // G(g-1,i+2)

//     g11 = _mm512_set1_pd(G[2 * (i + 2) + g * ldg]);     // G(g,i+2)
//     s11 = _mm512_set1_pd(G[2 * (i + 2) + g * ldg + 1]); // G(g,i+2)

//     g12 = _mm512_set1_pd(G[2 * (i + 2) + (g + 1) * ldg]);     // G(g+1,i+2)
//     s12 = _mm512_set1_pd(G[2 * (i + 2) + (g + 1) * ldg + 1]); // G(g+1,i+2)

//     /*Loop*/
//     for (int j = 0; j < m_iter; j++, v0 += 8, v1 += 8, v2 += 8, v3 += 8, v4 += 8, v5 += 8, v6 += 8)
//     {
//         v0_vec = _mm512_loadu_pd(v0);
//         v1_vec = _mm512_loadu_pd(v1);
//         v2_vec = _mm512_loadu_pd(v2);
//         v3_vec = _mm512_loadu_pd(v3);
//         v4_vec = _mm512_loadu_pd(v4);

//         /*G(g,i)*/
//         tmp_vec = v0_vec;
//         v0_vec = _mm512_add_pd(_mm512_mul_pd(g1, tmp_vec), _mm512_mul_pd(s1, v1_vec));
//         v1_vec = _mm512_sub_pd(_mm512_mul_pd(g1, v1_vec), _mm512_mul_pd(s1, tmp_vec));

//         /*G(g+1,i)*/
//         tmp_vec = v1_vec;
//         v1_vec = _mm512_add_pd(_mm512_mul_pd(g2, tmp_vec), _mm512_mul_pd(s2, v2_vec));
//         v2_vec = _mm512_sub_pd(_mm512_mul_pd(g2, v2_vec), _mm512_mul_pd(s2, tmp_vec));

//         /*G(g+2,i)*/
//         tmp_vec = v2_vec;
//         v2_vec = _mm512_add_pd(_mm512_mul_pd(g3, tmp_vec), _mm512_mul_pd(s3, v3_vec));
//         v3_vec = _mm512_sub_pd(_mm512_mul_pd(g3, v3_vec), _mm512_mul_pd(s3, tmp_vec));

//         /*G(g+3,i)*/
//         tmp_vec = v3_vec;
//         v3_vec = _mm512_add_pd(_mm512_mul_pd(g4, tmp_vec), _mm512_mul_pd(s4, v4_vec));
//         v4_vec = _mm512_sub_pd(_mm512_mul_pd(g4, v4_vec), _mm512_mul_pd(s4, tmp_vec));

//         /*G(g-1,i+1)*/
//         _mm512_storeu_pd(v4, v4_vec);
//         v4_vec = _mm512_loadu_pd(v5);
//         tmp_vec = v4_vec;
//         v4_vec = _mm512_add_pd(_mm512_mul_pd(g5, tmp_vec), _mm512_mul_pd(s5, v0_vec));
//         v0_vec = _mm512_sub_pd(_mm512_mul_pd(g5, v0_vec), _mm512_mul_pd(s5, tmp_vec));

//         /*G(g,i+1)*/
//         tmp_vec = v0_vec;
//         v0_vec = _mm512_add_pd(_mm512_mul_pd(g6, tmp_vec), _mm512_mul_pd(s6, v1_vec));
//         v1_vec = _mm512_sub_pd(_mm512_mul_pd(g6, v1_vec), _mm512_mul_pd(s6, tmp_vec));

//         /*G(g+1,i+1)*/
//         tmp_vec = v1_vec;
//         v1_vec = _mm512_add_pd(_mm512_mul_pd(g7, tmp_vec), _mm512_mul_pd(s7, v2_vec));
//         v2_vec = _mm512_sub_pd(_mm512_mul_pd(g7, v2_vec), _mm512_mul_pd(s7, tmp_vec));

//         /*G(g+2,i+1)*/
//         tmp_vec = v2_vec;
//         v2_vec = _mm512_add_pd(_mm512_mul_pd(g8, tmp_vec), _mm512_mul_pd(s8, v3_vec));
//         v3_vec = _mm512_sub_pd(_mm512_mul_pd(g8, v3_vec), _mm512_mul_pd(s8, tmp_vec));

//         /*G(g-2,i+2)*/
//         _mm512_storeu_pd(v3, v3_vec);
//         v3_vec = _mm512_loadu_pd(v6);
//         tmp_vec = v3_vec;
//         v3_vec = _mm512_add_pd(_mm512_mul_pd(g9, tmp_vec), _mm512_mul_pd(s9, v4_vec));
//         v4_vec = _mm512_sub_pd(_mm512_mul_pd(g9, v4_vec), _mm512_mul_pd(s9, tmp_vec));

//         /*G(g-1,i+2)*/
//         tmp_vec = v4_vec;
//         v4_vec = _mm512_add_pd(_mm512_mul_pd(g10, tmp_vec), _mm512_mul_pd(s10, v0_vec));
//         v0_vec = _mm512_sub_pd(_mm512_mul_pd(g10, v0_vec), _mm512_mul_pd(s10, tmp_vec));

//         /*G(g,i+2)*/
//         tmp_vec = v0_vec;
//         v0_vec = _mm512_add_pd(_mm512_mul_pd(g11, tmp_vec), _mm512_mul_pd(s11, v1_vec));
//         v1_vec = _mm512_sub_pd(_mm512_mul_pd(g11, v1_vec), _mm512_mul_pd(s11, tmp_vec));

//         /*G(g+1,i+2)*/
//         tmp_vec = v1_vec;
//         v1_vec = _mm512_add_pd(_mm512_mul_pd(g12, tmp_vec), _mm512_mul_pd(s12, v2_vec));
//         v2_vec = _mm512_sub_pd(_mm512_mul_pd(g12, v2_vec), _mm512_mul_pd(s12, tmp_vec));

//         _mm512_storeu_pd(v6, v3_vec);
//         _mm512_storeu_pd(v5, v4_vec);
//         _mm512_storeu_pd(v0, v0_vec);
//         _mm512_storeu_pd(v1, v1_vec);
//         _mm512_storeu_pd(v2, v2_vec);
//     }
//     if (m_left > 0)
//     {
//         __mmask8 mask = (__mmask8)(255 >> (8 - m_left));
//         v0_vec = _mm512_maskz_loadu_pd(mask, v0);
//         v1_vec = _mm512_maskz_loadu_pd(mask, v1);
//         v2_vec = _mm512_maskz_loadu_pd(mask, v2);
//         v3_vec = _mm512_maskz_loadu_pd(mask, v3);
//         v4_vec = _mm512_maskz_loadu_pd(mask, v4);
//         /*G(g,i)*/
//         tmp_vec = v0_vec;
//         v0_vec = _mm512_add_pd(_mm512_mul_pd(g1, tmp_vec), _mm512_mul_pd(s1, v1_vec));
//         v1_vec = _mm512_sub_pd(_mm512_mul_pd(g1, v1_vec), _mm512_mul_pd(s1, tmp_vec));

//         /*G(g+1,i)*/
//         tmp_vec = v1_vec;
//         v1_vec = _mm512_add_pd(_mm512_mul_pd(g2, tmp_vec), _mm512_mul_pd(s2, v2_vec));
//         v2_vec = _mm512_sub_pd(_mm512_mul_pd(g2, v2_vec), _mm512_mul_pd(s2, tmp_vec));

//         /*G(g+2,i)*/
//         tmp_vec = v2_vec;
//         v2_vec = _mm512_add_pd(_mm512_mul_pd(g3, tmp_vec), _mm512_mul_pd(s3, v3_vec));
//         v3_vec = _mm512_sub_pd(_mm512_mul_pd(g3, v3_vec), _mm512_mul_pd(s3, tmp_vec));

//         /*G(g+3,i)*/
//         tmp_vec = v3_vec;
//         v3_vec = _mm512_add_pd(_mm512_mul_pd(g4, tmp_vec), _mm512_mul_pd(s4, v4_vec));
//         v4_vec = _mm512_sub_pd(_mm512_mul_pd(g4, v4_vec), _mm512_mul_pd(s4, tmp_vec));

//         /*G(g-1,i+1)*/
//         _mm512_mask_storeu_pd(v4, mask, v4_vec);
//         v4_vec = _mm512_maskz_loadu_pd(mask, v5);
//         tmp_vec = v4_vec;
//         v4_vec = _mm512_add_pd(_mm512_mul_pd(g5, tmp_vec), _mm512_mul_pd(s5, v0_vec));
//         v0_vec = _mm512_sub_pd(_mm512_mul_pd(g5, v0_vec), _mm512_mul_pd(s5, tmp_vec));

//         /*G(g,i+1)*/
//         tmp_vec = v0_vec;
//         v0_vec = _mm512_add_pd(_mm512_mul_pd(g6, tmp_vec), _mm512_mul_pd(s6, v1_vec));
//         v1_vec = _mm512_sub_pd(_mm512_mul_pd(g6, v1_vec), _mm512_mul_pd(s6, tmp_vec));

//         /*G(g+1,i+1)*/
//         tmp_vec = v1_vec;
//         v1_vec = _mm512_add_pd(_mm512_mul_pd(g7, tmp_vec), _mm512_mul_pd(s7, v2_vec));
//         v2_vec = _mm512_sub_pd(_mm512_mul_pd(g7, v2_vec), _mm512_mul_pd(s7, tmp_vec));

//         /*G(g+2,i+1)*/
//         tmp_vec = v2_vec;
//         v2_vec = _mm512_add_pd(_mm512_mul_pd(g8, tmp_vec), _mm512_mul_pd(s8, v3_vec));
//         v3_vec = _mm512_sub_pd(_mm512_mul_pd(g8, v3_vec), _mm512_mul_pd(s8, tmp_vec));

//         /*G(g-2,i+2)*/
//         _mm512_mask_storeu_pd(v3, mask, v3_vec);
//         v3_vec = _mm512_maskz_loadu_pd(mask, v6);
//         tmp_vec = v3_vec;
//         v3_vec = _mm512_add_pd(_mm512_mul_pd(g9, tmp_vec), _mm512_mul_pd(s9, v4_vec));
//         v4_vec = _mm512_sub_pd(_mm512_mul_pd(g9, v4_vec), _mm512_mul_pd(s9, tmp_vec));

//         /*G(g-1,i+2)*/
//         tmp_vec = v4_vec;
//         v4_vec = _mm512_add_pd(_mm512_mul_pd(g10, tmp_vec), _mm512_mul_pd(s10, v0_vec));
//         v0_vec = _mm512_sub_pd(_mm512_mul_pd(g10, v0_vec), _mm512_mul_pd(s10, tmp_vec));

//         /*G(g,i+2)*/
//         tmp_vec = v0_vec;
//         v0_vec = _mm512_add_pd(_mm512_mul_pd(g11, tmp_vec), _mm512_mul_pd(s11, v1_vec));
//         v1_vec = _mm512_sub_pd(_mm512_mul_pd(g11, v1_vec), _mm512_mul_pd(s11, tmp_vec));

//         /*G(g+1,i+2)*/
//         tmp_vec = v1_vec;
//         v1_vec = _mm512_add_pd(_mm512_mul_pd(g12, tmp_vec), _mm512_mul_pd(s12, v2_vec));
//         v2_vec = _mm512_sub_pd(_mm512_mul_pd(g12, v2_vec), _mm512_mul_pd(s12, tmp_vec));

//         _mm512_mask_storeu_pd(v6, mask, v3_vec);
//         _mm512_mask_storeu_pd(v5, mask, v4_vec);
//         _mm512_mask_storeu_pd(v0, mask, v0_vec);
//         _mm512_mask_storeu_pd(v1, mask, v1_vec);
//         _mm512_mask_storeu_pd(v2, mask, v2_vec);
//     }
//     // for (int j = 0; j < m_left; j++, v0 += 1, v1 += 1, v2 += 1, v3 += 1, v4 += 1, v5 += 1, v6 += 1)
//     // {
//     //     /*G(g,i)*/
//     //     double tmp0 = *v0;
//     //     *v0 = G[2 * i + g * ldg] * tmp0 + G[2 * i + g * ldg + 1] * (*v1);
//     //     *v1 = G[2 * i + g * ldg] * (*v1) - G[2 * i + g * ldg + 1] * tmp0;

//     //     /*G(g+1,i)*/
//     //     double tmp1 = *v1;
//     //     *v1 = G[2 * i + (g + 1) * ldg] * tmp1 + G[2 * i + (g + 1) * ldg + 1] * (*v2);
//     //     *v2 = G[2 * i + (g + 1) * ldg] * (*v2) - G[2 * i + (g + 1) * ldg + 1] * tmp1;

//     //     /*G(g+2,i)*/
//     //     double tmp2 = *v2;
//     //     *v2 = G[2 * i + (g + 2) * ldg] * tmp2 + G[2 * i + (g + 2) * ldg + 1] * (*v3);
//     //     *v3 = G[2 * i + (g + 2) * ldg] * (*v3) - G[2 * i + (g + 2) * ldg + 1] * tmp2;

//     //     /*G(g+3,i)*/
//     //     double tmp3 = *v3;
//     //     *v3 = G[2 * i + (g + 3) * ldg] * tmp3 + G[2 * i + (g + 3) * ldg + 1] * (*v4);
//     //     *v4 = G[2 * i + (g + 3) * ldg] * (*v4) - G[2 * i + (g + 3) * ldg + 1] * tmp3;

//     //     /*G(g-1,i+1)*/
//     //     double tmp5 = *v5;
//     //     *v5 = G[2 * (i + 1) + (g - 1) * ldg] * tmp5 + G[2 * (i + 1) + (g - 1) * ldg + 1] * (*v0);
//     //     *v0 = G[2 * (i + 1) + (g - 1) * ldg] * (*v0) - G[2 * (i + 1) + (g - 1) * ldg + 1] * tmp5;

//     //     /*G(g,i+1)*/
//     //     double tmp0_1 = *v0;
//     //     *v0 = G[2 * (i + 1) + g * ldg] * tmp0_1 + G[2 * (i + 1) + g * ldg + 1] * (*v1);
//     //     *v1 = G[2 * (i + 1) + g * ldg] * (*v1) - G[2 * (i + 1) + g * ldg + 1] * tmp0_1;

//     //     /*G(g+1,i+1)*/
//     //     double tmp1_1 = *v1;
//     //     *v1 = G[2 * (i + 1) + (g + 1) * ldg] * tmp1_1 + G[2 * (i + 1) + (g + 1) * ldg + 1] * (*v2);
//     //     *v2 = G[2 * (i + 1) + (g + 1) * ldg] * (*v2) - G[2 * (i + 1) + (g + 1) * ldg + 1] * tmp1_1;

//     //     /*G(g+2,i+1)*/
//     //     double tmp2_1 = *v2;
//     //     *v2 = G[2 * (i + 1) + (g + 2) * ldg] * tmp2_1 + G[2 * (i + 1) + (g + 2) * ldg + 1] * (*v3);
//     //     *v3 = G[2 * (i + 1) + (g + 2) * ldg] * (*v3) - G[2 * (i + 1) + (g + 2) * ldg + 1] * tmp2_1;

//     //     /*G(g-2,i+2)*/
//     //     double tmp6 = *v6;
//     //     *v6 = G[2 * (i + 2) + (g - 2) * ldg] * tmp6 + G[2 * (i + 2) + (g - 2) * ldg + 1] * (*v5);
//     //     *v5 = G[2 * (i + 2) + (g - 2) * ldg] * (*v5) - G[2 * (i + 2) + (g - 2) * ldg + 1] * tmp6;

//     //     /*G(g-1,i+2)*/
//     //     double tmp5_1 = *v5;
//     //     *v5 = G[2 * (i + 2) + (g - 1) * ldg] * tmp5_1 + G[2 * (i + 2) + (g - 1) * ldg + 1] * (*v0);
//     //     *v0 = G[2 * (i + 2) + (g - 1) * ldg] * (*v0) - G[2 * (i + 2) + (g - 1) * ldg + 1] * tmp5_1;

//     //     /*G(g,i+2)*/
//     //     double tmp0_2 = *v0;
//     //     *v0 = G[2 * (i + 2) + g * ldg] * tmp0_2 + G[2 * (i + 2) + g * ldg + 1] * (*v1);
//     //     *v1 = G[2 * (i + 2) + g * ldg] * (*v1) - G[2 * (i + 2) + g * ldg + 1] * tmp0_2;

//     //     /*G(g+1,i+2)*/
//     //     double tmp1_2 = *v1;
//     //     *v1 = G[2 * (i + 2) + (g + 1) * ldg] * tmp1_2 + G[2 * (i + 2) + (g + 1) * ldg + 1] * (*v2);
//     //     *v2 = G[2 * (i + 2) + (g + 1) * ldg] * (*v2) - G[2 * (i + 2) + (g + 1) * ldg + 1] * tmp1_2;
//     // }
// }

void applywave_fusing_auto(int k, int m, int n, double *G, double *V, int ldv, int ldg, int mx, int my)
{
    if (n < k || k == 1)
    {
        applysingle_avx(k, m, n, G, V, ldv, ldg);
    }

    for (int j = -mx + 1; j < n + k; j += mx)
    {
        for (int i = 0, g = j; i < k; i += my, g -= my)
        {

            if (((g - my + 1 >= 0) && (g + mx - 1 < n - 1)) && (i + my - 1 < k))
            {
                // (g,i) starting point

                {
                    applywavemx2_avx_auto(m, V, G, ldv, ldg, g, i);
                }
            }
            else
            {

                for (int ii = i; ii < i + my && ii < k; ii++)
                {
                    int gg_start = g - (ii - i);
                    if (gg_start < 0)
                    {
                        gg_start = 0;
                    }

                    for (int gg = gg_start; gg < g + mx - (ii - i) && gg < n - 1; gg++)
                    {
                        //  printf("B:: %3d %3d\n", gg, ii);
                        /* only one left G(gg,ii)*/
                        double gamma = G[2 * ii + gg * ldg];
                        double sigma = G[2 * ii + gg * ldg + 1];
                        double *v = &V[gg * ldv];
                        double *v1 = &V[(gg + 1) * ldv];
                        applywavemx2_avx(m, gamma, sigma, v, v1);
                        // printf("%d,%d\n",gg,ii);
                    }
                }
            }
        }
    }
}

void dmatrix_vector_multiply_mt_auto(int k, int m, int n, double *g, double *v, int ldv, int ldg, int mx, int my)
{
    // #pragma omp parallel
    //     {

    //         int nt = omp_get_num_threads();
    //         int id = omp_get_thread_num();
    //         // split m
    //         int bm = (m + nt - 1) / nt;
    //         // bm = (bm+3)/4*4;
    //         bm = (bm + 7) / 8 * 8;
    //         int mbegin = bm * id < m ? bm * id : m;
    //         int mend = bm * (id + 1) < m ? bm * (id + 1) : m;

    //         // printf("%d %d %d\n",mbegin,mend,mend-mbegin);

    //         if (mend > mbegin)
    //         {

    //             applywave_fusing_auto(k, mend - mbegin, n, g, v + mbegin, ldv, ldg, mx, my);
    //         }
    //     }
    int bm = 128;
    int mend = 0;
    for (int mbegin = 0; mbegin < m; mbegin += bm)
    {
        if (m - mbegin < bm)
        {
            applywave_fusing_auto(k, m - mbegin, n, g, v + mbegin, ldv, ldg, mx, my);
        }
        else
        {
            applywave_fusing_auto(k, bm, n, g, v + mbegin, ldv, ldg, mx, my);
        }
    }
}
void dmatrix_vector_multiply_mt_op(int k, int m, int n, double *g, double *v, int ldv, int ldg, int mx, int my)
{
#pragma omp parallel
    {

        int nt = omp_get_num_threads();
        int id = omp_get_thread_num();
        // split m
        int bm = (m + nt - 1) / nt;
        // bm = (bm+3)/4*4;
        bm = (bm + 7) / 8 * 8;
        int mbegin = bm * id < m ? bm * id : m;
        int mend = bm * (id + 1) < m ? bm * (id + 1) : m;

        // printf("%d %d %d\n",mbegin,mend,mend-mbegin);

        if (mend > mbegin)
        {

            dmatrix_vector_multiply_mt_auto(k, mend - mbegin, n, g, v + mbegin, ldv, ldg, mx, my);
        }
    }
}
void dmatrix_vector_multiply_mt_Auto(int k, int m, int n, double *g, double *v, int ldv, int ldg, int mx, int my)
{
#pragma omp parallel
    {

        int nt = omp_get_num_threads();
        int id = omp_get_thread_num();
        // split m
        int bm = (m + nt - 1) / nt;
        // bm = (bm+3)/4*4;
        bm = (bm + 7) / 8 * 8;
        int mbegin = bm * id < m ? bm * id : m;
        int mend = bm * (id + 1) < m ? bm * (id + 1) : m;

        // printf("%d %d %d\n",mbegin,mend,mend-mbegin);

        if (mend > mbegin)
        {

            applywave_fusing_auto(k, mend - mbegin, n, g, v + mbegin, ldv, ldg, mx, my);
        }
    }
}
void dmatrix_vector_multiply_mt_avx(int k, int m, int n, double *g, double *v, int ldv, int ldg)
{
#pragma omp parallel
    {

        int nt = omp_get_num_threads();
        int id = omp_get_thread_num();
        // split m
        int bm = (m + nt - 1) / nt;
        // bm = (bm+3)/4*4;
        bm = (bm + 7) / 8 * 8;
        int mbegin = bm * id < m ? bm * id : m;
        int mend = bm * (id + 1) < m ? bm * (id + 1) : m;

        // printf("%d %d %d\n",mbegin,mend,mend-mbegin);

        if (mend > mbegin)
        {

            applywave_avx(k, mend - mbegin, n, g, v + mbegin, ldv, ldg);
        }
    }
}

double *copyMatrix(double *v, int m, int n, int ldv)
{
    double *tmp = (double *)malloc(sizeof(double) * ldv * n);
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            tmp[i + j * ldv] = v[i + j * ldv];
        }
    }
    return tmp;
}
int Check(double *v, double *vc, int m, int n, int ldv)
{
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < m; i++)
        {
            // if ((v[i + j * ldv] != vc[i + j * ldv]) > EPSILON)
            if (fabs(v[i + j * ldv] - vc[i + j * ldv]) > 1e-10)
            {
                // printf("%3d %3d %f %f\n", i, j, v[i + j * ldv], vc[i + j * ldv]);
                return 0;
            }
        }
    }
    return 1;
}
int main(int argc, char const *argv[])
{
    /* code */
    /* 2x2 fusing*/
    int m = atoi(argv[1]); // ROW
    int n = atoi(argv[1]);
    int k = atoi(argv[2]);
    int mx = atoi(argv[3]);
    int my = atoi(argv[4]);

    int ldv = m;     // >= m
    int ldg = 2 * k; // >= k
    if ((ldv % (4096 / 8)) == 0)
        ldv += 16;
    if ((ldg % (4096 / 8)) == 0)
        ldg += 16;

    /* code */
    double *v = dmatrix(m, n, ldv);
    double *g = dmatrix(2 * k, n - 1, ldg);
    double *cv; // = dmatrix(m, n, ldv);

    drandomM(m, n, v, ldv);
    drandomG(k, n - 1, g, ldg);
    // cv = copyMatrix(v, m, n, ldv);
    for (int i = 0; i < 5; i++)
    {

       
        double x = flush_cache(i64time() * 1e-9);
        long long int t1 = i64time();
        /*fusing*/
        dmatrix_vector_multiply_mt_auto(k, m, n, g, v, ldv, ldg, mx, my);
        long long int t2 = i64time();

        dmatrix_vector_multiply_mt_avx(k, m, n, g, cv, ldv, ldg);
        printf("%d\n", Check(v, cv, m, n, ldv));

        // double time1 = (t2 - t1) * 1e-9;
        // double flop = 6.0 * m * (n - 1) * k;
        // printf("%dX%d %d %d %f %f %f\n", mx, my, n, k, (flop / time1) * 1e-9, time1, x);
    }

    freedmatrix(v, m, n, ldv);
    freedmatrix(g, 2 * k, n - 1, ldg);

    return 0;
}

// gcc -O3  -march=native 15.c -fopenmp -lm -o 15
// OMP_NUM_THREADS=8 OMP_PLACES=0:8:2 ./15
