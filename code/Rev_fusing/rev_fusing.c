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
#include "apply_rev_avx.h"


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
void applygmx2(int m, double gamma, double sigma, double *x, double *y)
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
void applysingle_avx(int k, int m, int n, double *g, double *v, int ldv, int ldg)
{
    for (int j = 0; j < k; j++)
    {
        for (int h = 0; h < n - 1; h++)
        {
            double gamma = g[2 * h + j * ldg];
            double sigma = g[2 * h + j * ldg + 1];
            double *V = v + h * ldv;
            double *V1 = v + (h + 1) * ldv;
            applygmx2(m, gamma, sigma, V, V1);
        }
    }
}

void Check_seq(int m, int n, double *V, double *V_seq, int ldv, int i)
{
    int count = 0;
    for (int j = 0; j < n; j++)
    {
        for (int x = i; x < i + m; x++)

        {
            printf("%f %f\n", V[x + j * ldv], V_seq[count]);
            count++;
        }
        printf("\n");
    }
}

void copy_seq_avx256(int m, int n, double *V, double *tmp, int ldv, int i)
{
    int count = 0;
    for (int y = 0; y < n; y++)
    {
        // _mm_prefetch((const char *)&V[i + y * ldv], _MM_HINT_T0);
        for (int j = 0; j < m; j += 4)
        {
            __m256d data = _mm256_load_pd(&V[i + y * ldv + j]);
            _mm256_store_pd(&tmp[count + j], data);
        }
        count += m;
    }
}


void recover_seq_avx256(int m, int n, double *V, double *V_seq, int ldv, int i)
{
    int count = 0;
    for (int y = 0; y < n; y++)
    {
        // _mm_prefetch((const char *)&V_seq[count], _MM_HINT_T0);
        for (int j = 0; j < m; j += 4)
        {
            __m256d data = _mm256_load_pd(&V_seq[count + j]);
            _mm256_store_pd(&V[i + y * ldv + j], data);
        }
        count += m;
    }
}


void creat_left_seq_avx256(int m, int n, double *V, double *V_left, int ldv, int i, int m_left)
{
    int count = 0;

    for (int y = 0; y < n; y++)
    {
        int base_index = y * ldv + i;
        // _mm_prefetch((const char *)&V[base_index], _MM_HINT_T0);
        for (int j = 0; j < m_left; j += 4)
        {
            __m256d data = _mm256_load_pd(&V[base_index + j]);
            _mm256_store_pd(&V_left[count + j], data);
        }
        memset(&V_left[count + m_left], 0, (m - m_left) * sizeof(double));
        count += m;
    }
}


void recover_seq_left_avx256(int m, int n, double *V, double *V_seq, int ldv, int i, int m_left)
{
    int count = 0;
    for (int y = 0; y < n; y++)
    {
        int base_index = y * ldv + i;
        // _mm_prefetch((const char *)&V_seq[count], _MM_HINT_T0);
        for (int j = 0; j < m_left; j += 4)
        {
            __m256d data = _mm256_load_pd(&V_seq[count + j]);
            _mm256_store_pd(&V[base_index + j], data);
        }
        count += m;
    }
}



void apply_rev_avx256_auto_mv_seq_ALL(int K, int m, int n, double *G, double *V, int ldv, int ldg, int my, int mv)
{
    int m_iter = m / (mv * 4);
    int m_left = m % (mv * 4);
    int M = m_iter * (mv * 4);
    double *v_seq = (double *)_mm_malloc(sizeof(double) * (mv * 4) * n, 64);
    double *v_seq_left = (double *)_mm_malloc(sizeof(double) * (mv * 4) * n, 64);
    // double *v_seq __attribute__((aligned(32))) = (double *)_mm_malloc(sizeof(double) * (mv * 4) * n, 64);
    // double *v_seq_left __attribute__((aligned(32))) = (double *)_mm_malloc(sizeof(double) * (mv * 4) * n, 64);

    for (int i = 0; i < M; i += (mv * 4))
    {
        copy_seq_avx256(mv * 4, n, V, v_seq, ldv, i);
        for (int k = 0; k < K; k += my)
        {
            _mm_prefetch((const char *)&G[k * ldg], _MM_HINT_T0);
            apply_rev_avx_mv_seq_avx256(k, m, n, G, v_seq, ldg);
        }
        recover_seq_avx256(mv * 4, n, V, v_seq, ldv, i);
    }
    _mm_free(v_seq);

    if (m_left != 0)
    {
        creat_left_seq_avx256(mv * 4, n, V, v_seq_left, ldv, M, m_left);
        for (int k = 0; k < K; k += my)
        {
            _mm_prefetch((const char *)&G[k * ldg], _MM_HINT_T0);
            apply_rev_avx_mv_seq_avx256(k, m, n, G, v_seq_left, ldg);
        }
        recover_seq_left_avx256(mv * 4, n, V, v_seq_left, ldv, M, m_left);
    }
    _mm_free(v_seq_left);
}

void dmatrix_vector_multiply_mt_rev_avx256_seq_ALL(int k, int m, int n, double *g, double *v, int ldv, int ldg, int my, int mv)
{
#pragma omp parallel
    {

        int nt = omp_get_num_threads();
        int id = omp_get_thread_num();

        int bm = (m + nt - 1) / nt;

        bm = (bm + mv * 4 - 1) / (mv * 4) * (mv * 4);

        int mbegin = bm * id < m ? bm * id : m;
        int mend = bm * (id + 1) < m ? bm * (id + 1) : m;

        apply_rev_avx256_auto_mv_seq_ALL(k, mend - mbegin, n, g, v + mbegin, ldv, ldg, my, mv);
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
        bm = (bm + 3) / 4 * 4;
        int mbegin = bm * id < m ? bm * id : m;
        int mend = bm * (id + 1) < m ? bm * (id + 1) : m;

        // printf("%d %d %d\n",mbegin,mend,mend-mbegin);

        if (mend > mbegin)
        {

            applysingle_avx(k, mend - mbegin, n, g, v + mbegin, ldv, ldg);
        }
    }
}

double *copyMatrix(double *v, int m, int n, int ldv)
{
    double *tmp = (double *)malloc(sizeof(double) * ldv * n);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
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

    int my = atoi(argv[3]);
    int mv = atoi(argv[4]);

    int ldv = m;           // >= m
    int ldg = 2 * (n - 1); // >= k
    // ldv = ceil((4*mv*8)/64) * 64;
    if ((ldv % (4096 / 4)) == 0)
        ldv += 16;
    if ((ldg % (4096 / 4)) == 0)
        ldg += 16;

    /* code */
    double *v = dmatrix(m, n, ldv);
    double *g = dmatrix(2 * (n - 1), k, ldg);
    double *cv; // = dmatrix(m, n, ldv);

    drandomM(m, n, v, ldv);
    drandomG(n - 1, k, g, ldg);
    cv = copyMatrix(v, m, n, ldv);
    // printf("%p¥n", cv); fflush(stdout);
    // ldv
    for (int i = 0; i < 1; i++)
    {

        double x = flush_cache(i64time() * 1e-9);
        long long int t1 = i64time();
        /*fusing*/
        // dmatrix_vector_multiply_mt_rev_avx(k, m, n, g, v, ldv, ldg, my, mv);
        // dmatrix_vector_multiply_mt_rev_avx_seq(k, m, n, g, v, ldv, ldg, my, mv);
        dmatrix_vector_multiply_mt_rev_avx256_seq_ALL(k, m, n, g, v, ldv, ldg, my, mv);
        long long int t2 = i64time();

        // dmatrix_vector_multiply_mt_avx(k, m, n, g, cv, ldv, ldg);
        // // applysingle_avx(k, m, n, g, v, ldv, ldg);
        // printf("%d %d %d %d\n", my, mv, n, Check(v, cv, m, n, ldv));

        double time1 = (t2 - t1) * 1e-9;
        double flop = 6.0 * m * (n - 1) * k;
        printf("%dX%d %d %d %f %f %f\n", my, mv, n, k, (flop / time1) * 1e-9, time1, x);
    }

    freedmatrix(v, m, n, ldv);
    freedmatrix(g, 2 * (n - 1), k, ldg);

    return 0;
}