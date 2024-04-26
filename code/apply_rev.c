#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <stdint.h>
#include <omp.h>
#include <pmmintrin.h>
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
void apply_rev_my2(int K, int m, int n, double *G, double *V, int ldv, int ldg)
{
    for (int i = 0; i < m; i++)
    {
        for (int k = 0; k < K; k+=2)
        {
            double v0,v1;
            v0 = V[i];
            v1 = V[i+ldv];
            double g0 = G[2*k];
            double s0 = G[2*k+1];
            double tmp = v0;
            v0 = g0 * tmp + s0 * v1;
            v1 = g0 * v1 - s0 * tmp;
            for (int g = 1; g < n - 1; g++)
            {
                // double gamma = G[2 * k + i * ldg];
                // double sigma = G[2 * k + i * ldg + 1];
                // double *v = &V[i * ldv];
                // double *v1 = &V[(i + 1) * ldv];
                // applywavemx2(m, gamma, sigma, v, v1);

                // double *xp = &V[i + g * ldv];
                // double *yp = &V[i + (g + 1) * ldv];
                double v2 = V[i + (g + 1) * ldv];
                double gamma = G[2 * k + g * ldg];
                double sigma = G[2 * k + g * ldg + 1];
                double tmp = v1;
                v1 = gamma * tmp + sigma * v2;
                v2 = gamma * v2 - sigma * tmp;
                double gamma1 = G[2 * (k+1) + (g-1) * ldg];
                double sigma1 = G[2 * (k + 1) + (g - 1) * ldg+1];
                double tmp1 = v0;
                v0 = gamma1 * tmp + sigma1 * v1;
                v1 = gamma1 * v1 - sigma1 * tmp;
                V[i+(g-1)*ldv]=v0;
                v0 = v1;
                v1 = v2;
            }
            double gend = G[2*(k+1)+(n-2)*ldg];
            double send = G[2 * (k + 1) + (n - 2) * ldg+1];
            tmp = v0;
            v0 = gend * tmp + send * v1;
            v1 = gend * v1 - send * tmp;
            V[i+(n-2)*ldv]= v0;
            V[i+(n-1)*ldv] = v1;
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
                // printf("%3d %3d %f %f\n", i, j, v[i + j * ldv], vc[i + j * ldv]);
                return 0;
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
    int k = 194;
    int ldv = m + 16;     // >= m
    int ldg = 2 * k + 16; // >= k

    double *v = dmatrix(m, n, ldv);
    double *g = dmatrix(2 * k, n - 1, ldg);
    double *vc = dmatrix(m, n, ldv);

    for (int i = 0; i < 1; i++)
    {
        /* code */

        // double *v = dmatrix(m, n, ldv);

        // apply_rev(k, m, n, g, vc, ldv, ldg);
        applyFrancis(k, m, n, g, v, ldv, ldg);
        apply_rev_my2(k, m, n, g, vc, ldv, ldg);
        printf("%d\n", Check(v, vc, m, n, ldv));

        // free(v);
    }
    free(g);
    free(vc);

    return 0;
}