#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <stdint.h>
#include <omp.h>
#include <pmmintrin.h>
#include <immintrin.h>
#include "apply_rev_avx.h"
int64_t i64time()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * (1000000000ull) + (int64_t)ts.tv_nsec;
}
double *dmatrix(int m, int n, int lda)
{
    assert(m > 0 && n > 0);
    assert(lda >= m);
    // double *ret = (double *)malloc(sizeof(double) * lda * n);
    double *ret = (double *)_mm_malloc(sizeof(double) * lda * n, 32);
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
void apply_rev(int K, int m, int n, double *G, double *V, int ldv, int ldg)
{
    for (int i = 0; i < m; i++)
    {
        for (int k = 0; k < K; k++)
        {
            for (int g = 0; g < n - 1; g++)
            {
                // double gamma = G[2 * k + i * ldg];
                // double sigma = G[2 * k + i * ldg + 1];
                // double *v = &V[i * ldv];
                // double *v1 = &V[(i + 1) * ldv];
                // applywavemx2(m, gamma, sigma, v, v1);

                double *xp = &V[i + g * ldv];
                double *yp = &V[i + (g + 1) * ldv];
                double gamma = G[2 * k + g * ldg];
                double sigma = G[2 * k + g * ldg + 1];
                double tmp = *xp;
                *xp = gamma * tmp + sigma * (*yp);
                *yp = gamma * (*yp) - sigma * tmp;
            }
        }
    }
}
void apply_rev_avx_auto_mv(int K, int m, int n, double *G, double *V, int ldv, int ldg, int my,int mv)
{
    for (int i = 0; i < m; i += mv * 4)
    {
        for (int k = 0; k < K; k += my)
        {
            apply_rev_avx_mv(k, m, n, G, V, ldv, ldg,i);
        }
    }
    
}
void apply_rev_av512_auto_mv(int K, int m, int n, double *G, double *V, int ldv, int ldg, int my,int mv)
{
    for (int i = 0; i < m; i += mv * 8)
    {
        for (int k = 0; k < K; k += my)
        {
           apply_rev_avx512_mv(k, m, n, G, V, ldv, ldg,i);
        }
    }
}
void apply_rec_my2_avx(int K, int m, int n, double *G, double *V, int ldv, int ldg)
{
    __m256d v0, v1, v2, gamma, sigma, tmp;
    for (int i = 0; i < m; i += 4)
    {
        for (int k = 0; k < K; k += 2)
        {

            v0 = _mm256_loadu_pd(&V[i]);
            v1 = _mm256_loadu_pd(&V[i + ldv]);
            /*G(k,0)*/
            gamma = _mm256_broadcast_sd(&G[2 * k]);
            sigma = _mm256_broadcast_sd(&G[2 * k + 1]);
            tmp = v0;
            v0 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v1));
            v1 = _mm256_sub_pd(_mm256_mul_pd(gamma, v1), _mm256_mul_pd(sigma, tmp));
            for (int g = 1; g < n - 1; g++)
            {
                v2 = _mm256_loadu_pd(&V[i + (g + 1) * ldv]);

                /*G(k,g)*/
                gamma = _mm256_broadcast_sd(&G[2 * k + g * ldg]);
                sigma = _mm256_broadcast_sd(&G[2 * k + g * ldg + 1]);
                tmp = v1;
                v1 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v2));
                v2 = _mm256_sub_pd(_mm256_mul_pd(gamma, v2), _mm256_mul_pd(sigma, tmp));

                /*G(k+1,g-1)*/
                gamma = _mm256_broadcast_sd(&G[2 * (k + 1) + (g - 1) * ldg]);
                sigma = _mm256_broadcast_sd(&G[2 * (k + 1) + (g - 1) * ldg + 1]);
                tmp = v0;
                v0 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v1));
                v1 = _mm256_sub_pd(_mm256_mul_pd(gamma, v1), _mm256_mul_pd(sigma, tmp));

                _mm256_storeu_pd(&V[i + (g - 1) * ldv], v0);
                v0 = v1;
                v1 = v2;
            }
            /*G(k+1,n-2)*/
            gamma = _mm256_broadcast_sd(&G[2 * (k + 1) + (n - 2) * ldg]);
            sigma = _mm256_broadcast_sd(&G[2 * (k + 1) + (n - 2) * ldg + 1]);
            tmp = v0;
            v0 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v1));
            v1 = _mm256_sub_pd(_mm256_mul_pd(gamma, v1), _mm256_mul_pd(sigma, tmp));
            _mm256_storeu_pd(&V[i + (n - 2) * ldv], v0);
            _mm256_storeu_pd(&V[i + (n - 1) * ldv], v1);
        }
    }
}
// void apply_rec_my2_avx_mv(int K, int m, int n, double *G, double *V, int ldv, int ldg)
// {
//     __m256d v0, v1, v2, gamma, sigma, tmp;
//     for (int i = 0; i < m; i += 4)
//     {
//         for (int k = 0; k < K; k += 2)
//         {

//             v0 = _mm256_loadu_pd(&V[i]);
//             v1 = _mm256_loadu_pd(&V[i + 4]);
//             v2 = _mm256_loadu_pd(&V[i + ldv]);
//             v3 = _mm256_loadu_pd(&V[i + ldv + 4]);
//             /*G(k,0)*/
//             gamma = _mm256_broadcast_sd(&G[2 * k]);
//             sigma = _mm256_broadcast_sd(&G[2 * k + 1]);

//             tmp = v0;
//             v0 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v2));
//             v2 = _mm256_sub_pd(_mm256_mul_pd(gamma, v2), _mm256_mul_pd(sigma, tmp));

//             tmp = v1;
//             v1 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v1));
//             v3 = _mm256_sub_pd(_mm256_mul_pd(gamma, v3), _mm256_mul_pd(sigma, tmp));

//             for (int g = 1; g < n - 1; g++)
//             {
//                 v20 = _mm256_loadu_pd(&V[i + (g + 1) * ldv]);
//                 v21 = _mm256_loadu_pd(&V[i + (g + 1) * ldv + 4]);

//                 /*G(k,g)*/
//                 gamma = _mm256_broadcast_sd(&G[2 * k + g * ldg]);
//                 sigma = _mm256_broadcast_sd(&G[2 * k + g * ldg + 1]);
//                 tmp = v1;
//                 v1 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v2));
//                 v2 = _mm256_sub_pd(_mm256_mul_pd(gamma, v2), _mm256_mul_pd(sigma, tmp));

//                 /*G(k+1,g-1)*/
//                 gamma = _mm256_broadcast_sd(&G[2 * (k + 1) + (g - 1) * ldg]);
//                 sigma = _mm256_broadcast_sd(&G[2 * (k + 1) + (g - 1) * ldg + 1]);
//                 tmp = v0;
//                 v0 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v1));
//                 v1 = _mm256_sub_pd(_mm256_mul_pd(gamma, v1), _mm256_mul_pd(sigma, tmp));

//                 _mm256_storeu_pd(&V[i + (g - 1) * ldv], v00);
//                 _mm256_storeu_pd(&V[i + (g - 1) * ldv + 4], v01);

//                 v00 = v10;
//                 v01 = v11;
//                 v10 = v20;
//                 v11 = v21;
//             }
//             /*G(k+1,n-2)*/
//             gamma = _mm256_broadcast_sd(&G[2 * (k + 1) + (n - 2) * ldg]);
//             sigma = _mm256_broadcast_sd(&G[2 * (k + 1) + (n - 2) * ldg + 1]);
//             tmp = v00;
//             v00 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v10));
//             v10 = _mm256_sub_pd(_mm256_mul_pd(gamma, v10), _mm256_mul_pd(sigma, tmp));

//             tmp = v01;
//             v01 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v11));
//             v11 = _mm256_sub_pd(_mm256_mul_pd(gamma, v11), _mm256_mul_pd(sigma, tmp));
//             _mm256_storeu_pd(&V[i + (n - 2) * ldv], v00);
//             _mm256_storeu_pd(&V[i + (n - 2) * ldv + 4], v01);
//             _mm256_storeu_pd(&V[i + (n - 1) * ldv], v10);
//             _mm256_storeu_pd(&V[i + (n - 1) * ldv + 4], v11);
//         }
//     }
// }
void apply_rev_my3_avx(int K, int m, int n, double *G, double *V, int ldv, int ldg)
{
    for (int i = 0; i < m; i += 4)
    {
        for (int k = 0; k < K; k += 3)
        {
            __m256d v0, v1, v2, v3, gamma, sigma, tmp;
            v0 = _mm256_loadu_pd(&V[i]);
            v1 = _mm256_loadu_pd(&V[i + ldv]);
            v2 = _mm256_loadu_pd(&V[i + 2 * ldv]);

            gamma = _mm256_broadcast_sd(&G[2 * k]);
            sigma = _mm256_broadcast_sd(&G[2 * k + 1]);
            tmp = v0;
            v0 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v1));
            v1 = _mm256_sub_pd(_mm256_mul_pd(gamma, v1), _mm256_mul_pd(sigma, tmp));

            gamma = _mm256_broadcast_sd(&G[2 * k + ldg]);
            sigma = _mm256_broadcast_sd(&G[2 * k + ldg + 1]);
            tmp = v1;
            v1 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v2));
            v2 = _mm256_sub_pd(_mm256_mul_pd(gamma, v2), _mm256_mul_pd(sigma, tmp));

            gamma = _mm256_broadcast_sd(&G[2 * (k + 1)]);
            sigma = _mm256_broadcast_sd(&G[2 * (k + 1) + 1]);
            tmp = v0;
            v0 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v1));
            v1 = _mm256_sub_pd(_mm256_mul_pd(gamma, v1), _mm256_mul_pd(sigma, tmp));

            for (int g = 2; g < n - 1; g++)
            {
                v3 = _mm256_loadu_pd(&V[i + (g + 1) * ldv]);
                /*G(k,g)*/
                gamma = _mm256_broadcast_sd(&G[2 * k + g * ldg]);
                sigma = _mm256_broadcast_sd(&G[2 * k + g * ldg + 1]);
                tmp = v2;
                v2 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v3));
                v3 = _mm256_sub_pd(_mm256_mul_pd(gamma, v3), _mm256_mul_pd(sigma, tmp));
                /*G(k+1,g-1)*/
                gamma = _mm256_broadcast_sd(&G[2 * (k + 1) + (g - 1) * ldg]);
                sigma = _mm256_broadcast_sd(&G[2 * (k + 1) + (g - 1) * ldg + 1]);
                tmp = v1;
                v1 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v2));
                v2 = _mm256_sub_pd(_mm256_mul_pd(gamma, v2), _mm256_mul_pd(sigma, tmp));
                /*G(k+2,g-2)*/
                gamma = _mm256_broadcast_sd(&G[2 * (k + 2) + (g - 2) * ldg]);
                sigma = _mm256_broadcast_sd(&G[2 * (k + 2) + (g - 2) * ldg + 1]);
                tmp = v0;
                v0 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v1));
                v1 = _mm256_sub_pd(_mm256_mul_pd(gamma, v1), _mm256_mul_pd(sigma, tmp));

                _mm256_storeu_pd(&V[i + (g - 2) * ldv], v0);

                v0 = v1;
                v1 = v2;
                v2 = v3;
            }
            /*G(k+1,n-2)*/
            gamma = _mm256_broadcast_sd(&G[2 * (k + 1) + (n - 2) * ldg]);
            sigma = _mm256_broadcast_sd(&G[2 * (k + 1) + (n - 2) * ldg + 1]);
            tmp = v1;
            v1 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v2));
            v2 = _mm256_sub_pd(_mm256_mul_pd(gamma, v2), _mm256_mul_pd(sigma, tmp));
            /*G(k+2,n-3)*/
            gamma = _mm256_broadcast_sd(&G[2 * (k + 2) + (n - 3) * ldg]);
            sigma = _mm256_broadcast_sd(&G[2 * (k + 2) + (n - 3) * ldg + 1]);
            tmp = v0;
            v0 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v1));
            v1 = _mm256_sub_pd(_mm256_mul_pd(gamma, v1), _mm256_mul_pd(sigma, tmp));
            /*G(k+2,n-2)*/
            gamma = _mm256_broadcast_sd(&G[2 * (k + 2) + (n - 2) * ldg]);
            sigma = _mm256_broadcast_sd(&G[2 * (k + 2) + (n - 2) * ldg + 1]);
            tmp = v1;
            v1 = _mm256_add_pd(_mm256_mul_pd(gamma, tmp), _mm256_mul_pd(sigma, v2));
            v2 = _mm256_sub_pd(_mm256_mul_pd(gamma, v2), _mm256_mul_pd(sigma, tmp));

            _mm256_storeu_pd(&V[i + (n - 3) * ldv], v0);
            _mm256_storeu_pd(&V[i + (n - 2) * ldv], v1);
            _mm256_storeu_pd(&V[i + (n - 1) * ldv], v2);
        }
    }
}
void apply_rev_my2(int K, int m, int n, double *G, double *V, int ldv, int ldg)
{

    for (int i = 0; i < m; i++)
    {
        double v0, v1, v2;
        for (int k = 0; k < K; k += 2)
        {
            /*starting with the singlg one Givens rotation*/
            v0 = V[i];
            v1 = V[i + ldv];

            /*paramater*/
            double gamma = G[2 * k];
            double sigma = G[2 * k + 1];

            double tmp;
            tmp = v0;
            v0 = gamma * tmp + sigma * v1;
            v1 = gamma * v1 - sigma * tmp;

            for (int g = 1; g < n - 1; g++)
            {
                v2 = V[i + (g + 1) * ldv];

                double gamma1 = G[2 * k + g * ldg];
                double sigma1 = G[2 * k + g * ldg + 1];

                double gamma2 = G[2 * (k + 1) + (g - 1) * ldg];
                double sigma2 = G[2 * (k + 1) + (g - 1) * ldg + 1];

                double tmp1;
                /*G(k,g)*/
                tmp1 = v1;
                v1 = gamma1 * tmp1 + sigma1 * v2;
                v2 = gamma1 * v2 - sigma1 * tmp1;

                /*G(k+1,g-1)*/
                tmp1 = v0;
                v0 = gamma2 * tmp1 + sigma2 * v1;
                v1 = gamma2 * v1 - sigma2 * tmp1;

                V[i + (g - 1) * ldv] = v0;
                v0 = v1;
                v1 = v2;
            }

            /*G(k+1,n-2)*/
            double gamma_end = G[2 * (k + 1) + (n - 2) * ldg];
            double sigma_end = G[2 * (k + 1) + (n - 2) * ldg + 1];

            tmp = v0;
            v0 = gamma_end * tmp + sigma_end * v1;
            v1 = gamma_end * v1 - sigma_end * tmp;
            V[i + (n - 2) * ldv] = v0;
            V[i + (n - 1) * ldv] = v1;
        }
    }
}
void apply_rev_my3(int K, int m, int n, double *G, double *V, int ldv, int ldg, int my)
{
    for (int i = 0; i < m; i++)
    {
        for (int k = 0; k < K; k += 3)
        {
            double v0, v1, v2;
            v0 = V[i];
            v1 = V[i + ldv];
            v2 = V[i + 2 * ldv];
            double g1, g2, g3, s1, s2, s3;
            g1 = G[2 * k];
            s1 = G[2 * k + 1];

            g2 = G[2 * k + ldg];
            s2 = G[2 * k + ldg + 1];

            g3 = G[2 * (k + 1)];
            s3 = G[2 * (k + 1) + 1];

            double tmp;
            /*G(k,0)*/
            tmp = v0;
            v0 = g1 * tmp + s1 * v1;
            v1 = g1 * v1 - s1 * tmp;

            tmp = v1;
            v1 = g2 * tmp + s2 * v2;
            v2 = g2 * v2 - s2 * tmp;

            tmp = v0;
            v0 = g3 * tmp + s3 * v1;
            v1 = g3 * v1 - s3 * tmp;

            for (int g = 2; g < n - 1; g++)
            {
                double v3 = V[i + (g + 1) * ldv];

                double gamma1, gamma2, gamma3;
                double sigma1, sigma2, sigma3;

                gamma1 = G[2 * k + g * ldg];
                sigma1 = G[2 * k + g * ldg + 1];

                gamma2 = G[2 * (k + 1) + (g - 1) * ldg];
                sigma2 = G[2 * (k + 1) + (g - 1) * ldg + 1];

                gamma3 = G[2 * (k + 2) + (g - 2) * ldg];
                sigma3 = G[2 * (k + 2) + (g - 2) * ldg + 1];

                /*G(k,g)*/
                tmp = v2;
                v2 = gamma1 * tmp + sigma1 * v3;
                v3 = gamma1 * v3 - sigma1 * tmp;

                /*G(k+1,g-1)*/
                tmp = v1;
                v1 = gamma2 * tmp + sigma2 * v2;
                v2 = gamma2 * v2 - sigma2 * tmp;

                /*G(k+2,g-2)*/
                tmp = v0;
                v0 = gamma3 * tmp + sigma3 * v1;
                v1 = gamma3 * v1 - sigma3 * tmp;

                V[i + (g - 2) * ldv] = v0;
                v0 = v1;
                v1 = v2;
                v2 = v3;
            }
            /*G(k+1,n-2)*/
            g1 = G[2 * (k + 1) + (n - 2) * ldg];
            s1 = G[2 * (k + 1) + (n - 2) * ldg + 1];

            /*G(k+2,n-3)*/
            g2 = G[2 * (k + 2) + (n - 3) * ldg];
            s2 = G[2 * (k + 2) + (n - 3) * ldg + 1];

            /*G(k+2,n-2)*/
            g3 = G[2 * (k + 2) + (n - 2) * ldg];
            s3 = G[2 * (k + 2) + (n - 2) * ldg + 1];

            tmp = v1;
            v1 = g1 * tmp + s1 * v2;
            v2 = g1 * v2 - s1 * tmp;

            tmp = v0;
            v0 = g2 * tmp + s2 * v1;
            v1 = g2 * v1 - s2 * tmp;

            tmp = v1;
            v1 = g3 * tmp + s3 * v2;
            v2 = g3 * v2 - s3 * tmp;

            V[i + (n - 3) * ldv] = v0;
            V[i + (n - 2) * ldv] = v1;
            V[i + (n - 1) * ldv] = v2;
        }
    }
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
                printf("%3d %3d %f %f\n", i, j, v[i + j * ldv], vc[i + j * ldv]);
                // return 0;
            }
        }
    }
    return 1;
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
void applywavemx2(int m, double gamma, double sigma, double *x, double *y)
{
    // applying a single Givens rotation
    double *restrict xp = x;
    double *restrict yp = y;
    double tmp = 0.0;
    if (gamma == 1.0)
    {
        return;
    }
    for (int i = 0; i < m; i++)
    {
        tmp = *xp;
        *xp = gamma * tmp + sigma * (*yp);
        *yp = gamma * (*yp) - sigma * tmp;
        xp += 1;
        yp += 1;
    }
}
void applyFrancis(int K, int m, int n, double *G, double *V, int ldv, int ldg)
{
    for (int k = 0; k < K; k++)
    {
        for (int i = 0; i < n - 1; i++)
        {
            double gamma = G[2 * k + i * ldg];
            double sigma = G[2 * k + i * ldg + 1];
            double *v = &V[i * ldv];
            double *v1 = &V[(i + 1) * ldv];
            applywavemx2(m, gamma, sigma, v, v1);
        }
    }
}
int main(int argc, char const *argv[])
{
    /* code */
    int m = atoi(argv[1]); // ROW
    int n = atoi(argv[1]);
    int k = 192;
    int ldv = m + 16;     // >= m
    int ldg = 2 * k + 16; // >= k

    double *v = dmatrix(m, n, ldv);
    double *g = dmatrix(2 * k, n - 1, ldg);
    double *vc;
    drandomM(m, n, v, ldv);
    drandomG(k, n - 1, g, ldg);

    vc = copyMatrix(v, m, n, ldv);

    for (int i = 0; i < 1; i++)
    {
        /* code */

        // apply_rev(k, m, n, g, v, ldv, ldg);
        // apply_rev_avx(k, m, n, g, vc, ldv, ldg);
        // applyFrancis(k, m, n, g, v, ldv, ldg);
        // apply_rev_my2(k, m, n, g, vc, ldv, ldg);
        // apply_rec_my2_avx(k, m, n, g, v, ldv, ldg);
        // apply_rev_my3_avx(k, m, n, g, v, ldv, ldg);
        apply_rev_avx_auto_mv(k, m, n, g, v, ldv, ldg, 3,2);
        apply_rev_av512_auto_mv(k, m, n, g, vc, ldv, ldg, 3,2);

        // apply_rev_my3(k, m, n, g, vc, ldv, ldg, 3);
        printf("%d\n", Check(v, vc, m, n, ldv));

        // free(v);
    }
    free(g);
    free(vc);

    return 0;
}
