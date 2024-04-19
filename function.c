#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
 #include "function.h" 


void applywavemx2_avx_2x3(int m, double *V, double *G, int ldv, int ldg, int g, int i)
{
  double *restrict v0 = &V[(g+0) * ldv];
  double *restrict v1 = &V[(g+1) * ldv];
  double *restrict v2 = &V[(g+2) * ldv];
  double *restrict v3 = &V[(g-1) * ldv];
  double *restrict v4 = &V[(g-2) * ldv];
__m512d  g1 ,g2 ,g3 ,g4 ,g5 ,g6 ; 
__m512d  s1 ,s2 ,s3 ,s4 ,s5 ,s6 ; 
__m512d  v0_vec ,v1_vec,v2_vec; 


int m_iter = m / 8;
int m_left = m % 8;


__m512d  tmp_vec;
  g1=_mm512_set1_pd(G[2 * (i+0) + (g+0) * ldg]); 
  s1=_mm512_set1_pd(G[2 * (i+0) + (g+0) * ldg+1]); 


  g2=_mm512_set1_pd(G[2 * (i+0) + (g+1) * ldg]); 
  s2=_mm512_set1_pd(G[2 * (i+0) + (g+1) * ldg+1]); 


  g3=_mm512_set1_pd(G[2 * (i+1) + (g+-1) * ldg]); 
  s3=_mm512_set1_pd(G[2 * (i+1) + (g+-1) * ldg+1]); 


  g4=_mm512_set1_pd(G[2 * (i+1) + (g+0) * ldg]); 
  s4=_mm512_set1_pd(G[2 * (i+1) + (g+0) * ldg+1]); 


  g5=_mm512_set1_pd(G[2 * (i+2) + (g+-2) * ldg]); 
  s5=_mm512_set1_pd(G[2 * (i+2) + (g+-2) * ldg+1]); 


  g6=_mm512_set1_pd(G[2 * (i+2) + (g+-1) * ldg]); 
  s6=_mm512_set1_pd(G[2 * (i+2) + (g+-1) * ldg+1]); 


 for (int j = 0; j < m_iter; j++,v0 += 8,v1 += 8,v2 += 8,v3 += 8,v4 += 8)
{
  v0_vec = _mm512_loadu_pd(v0);
  v1_vec = _mm512_loadu_pd(v1);
  v2_vec = _mm512_loadu_pd(v2);
  tmp_vec = v0_vec;
  v0_vec= _mm512_add_pd(_mm512_mul_pd(g1, tmp_vec), _mm512_mul_pd(s1, v1_vec));
  v1_vec = _mm512_sub_pd(_mm512_mul_pd(g1, v1_vec), _mm512_mul_pd(s1, tmp_vec));


  tmp_vec = v1_vec;
  v1_vec= _mm512_add_pd(_mm512_mul_pd(g2, tmp_vec), _mm512_mul_pd(s2, v2_vec));
  v2_vec = _mm512_sub_pd(_mm512_mul_pd(g2, v2_vec), _mm512_mul_pd(s2, tmp_vec));


  _mm512_storeu_pd(v2, v2_vec);
  v2_vec = _mm512_loadu_pd(v3);
  tmp_vec = v2_vec;
  v2_vec= _mm512_add_pd(_mm512_mul_pd(g3, tmp_vec), _mm512_mul_pd(s3, v0_vec));
  v0_vec = _mm512_sub_pd(_mm512_mul_pd(g3, v0_vec), _mm512_mul_pd(s3, tmp_vec));


  tmp_vec = v0_vec;
  v0_vec= _mm512_add_pd(_mm512_mul_pd(g4, tmp_vec), _mm512_mul_pd(s4, v1_vec));
  v1_vec = _mm512_sub_pd(_mm512_mul_pd(g4, v1_vec), _mm512_mul_pd(s4, tmp_vec));


  _mm512_storeu_pd(v1, v1_vec);
  v1_vec = _mm512_loadu_pd(v4);
  tmp_vec = v1_vec;
  v1_vec= _mm512_add_pd(_mm512_mul_pd(g5, tmp_vec), _mm512_mul_pd(s5, v2_vec));
  v2_vec = _mm512_sub_pd(_mm512_mul_pd(g5, v2_vec), _mm512_mul_pd(s5, tmp_vec));


  tmp_vec = v2_vec;
  v2_vec= _mm512_add_pd(_mm512_mul_pd(g6, tmp_vec), _mm512_mul_pd(s6, v0_vec));
  v0_vec = _mm512_sub_pd(_mm512_mul_pd(g6, v0_vec), _mm512_mul_pd(s6, tmp_vec));


  _mm512_storeu_pd(v4, v1_vec);
  _mm512_storeu_pd(v3, v2_vec);
  _mm512_storeu_pd(v0, v0_vec);
}
  if (m_left > 0)
{
 __mmask8 mask = (__mmask8)(255 >> (8 - m_left));


  v0_vec = _mm512_maskz_loadu_pd(mask, v0);
  v1_vec = _mm512_maskz_loadu_pd(mask, v1);
  v2_vec = _mm512_maskz_loadu_pd(mask, v2);


  tmp_vec = v0_vec;
  v0_vec = _mm512_add_pd(_mm512_mul_pd(g1, tmp_vec), _mm512_mul_pd(v1_vec, v0_vec));
  v1_vec = _mm512_sub_pd(_mm512_mul_pd(g1, v1_vec), _mm512_mul_pd(s1, tmp_vec));


  tmp_vec = v1_vec;
  v1_vec = _mm512_add_pd(_mm512_mul_pd(g2, tmp_vec), _mm512_mul_pd(v2_vec, v1_vec));
  v2_vec = _mm512_sub_pd(_mm512_mul_pd(g2, v2_vec), _mm512_mul_pd(s2, tmp_vec));


  _mm512_mask_storeu_pd(v2, mask, v2_vec);
  v1_vec = _mm512_maskz_loadu_pd(mask, v3);
  tmp_vec = v2_vec;
  v2_vec = _mm512_add_pd(_mm512_mul_pd(g3, tmp_vec), _mm512_mul_pd(v0_vec, v2_vec));
  v0_vec = _mm512_sub_pd(_mm512_mul_pd(g3, v0_vec), _mm512_mul_pd(s3, tmp_vec));


  tmp_vec = v0_vec;
  v0_vec = _mm512_add_pd(_mm512_mul_pd(g4, tmp_vec), _mm512_mul_pd(v1_vec, v0_vec));
  v1_vec = _mm512_sub_pd(_mm512_mul_pd(g4, v1_vec), _mm512_mul_pd(s4, tmp_vec));


  _mm512_mask_storeu_pd(v1, mask, v1_vec);
  v1_vec = _mm512_maskz_loadu_pd(mask, v4);
  tmp_vec = v1_vec;
  v1_vec = _mm512_add_pd(_mm512_mul_pd(g5, tmp_vec), _mm512_mul_pd(v2_vec, v1_vec));
  v2_vec = _mm512_sub_pd(_mm512_mul_pd(g5, v2_vec), _mm512_mul_pd(s5, tmp_vec));


  tmp_vec = v2_vec;
  v2_vec = _mm512_add_pd(_mm512_mul_pd(g6, tmp_vec), _mm512_mul_pd(v0_vec, v2_vec));
  v0_vec = _mm512_sub_pd(_mm512_mul_pd(g6, v0_vec), _mm512_mul_pd(s6, tmp_vec));


  _mm512_mask_storeu_pd(v4, mask, v1_vec);
  _mm512_mask_storeu_pd(v3, mask, v2_vec);
  _mm512_mask_storeu_pd(v0, mask, v0_vec);
}
}
