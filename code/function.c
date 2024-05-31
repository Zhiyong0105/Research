#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
 #include "function.h" 


void applywavemx2_avx_auto(int m, double *V, double *G, int ldv, int ldg, int g, int i)
{
  double *restrict v0 = &V[(g+0) * ldv];
  double *restrict v1 = &V[(g+1) * ldv];
  double *restrict v2 = &V[(g+2) * ldv];
  double *restrict v3 = &V[(g+3) * ldv];
  double *restrict v4 = &V[(g+4) * ldv];
  double *restrict v5 = &V[(g+5) * ldv];
  double *restrict v6 = &V[(g+6) * ldv];
  double *restrict v7 = &V[(g-1) * ldv];
  double *restrict v8 = &V[(g-2) * ldv];
  double *restrict v9 = &V[(g-3) * ldv];
  double *restrict v10 = &V[(g-4) * ldv];
  double *restrict v11 = &V[(g-5) * ldv];
__m512d  g1 ,g2 ,g3 ,g4 ,g5 ,g6 ,g7 ,g8 ,g9 ,g10 ,g11 ,g12 ,g13 ,g14 ,g15 ,g16 ,g17 ,g18 ,g19 ,g20 ,g21 ,g22 ,g23 ,g24 ,g25 ,g26 ,g27 ,g28 ,g29 ,g30 ,g31 ,g32 ,g33 ,g34 ,g35 ,g36 ; 
__m512d  s1 ,s2 ,s3 ,s4 ,s5 ,s6 ,s7 ,s8 ,s9 ,s10 ,s11 ,s12 ,s13 ,s14 ,s15 ,s16 ,s17 ,s18 ,s19 ,s20 ,s21 ,s22 ,s23 ,s24 ,s25 ,s26 ,s27 ,s28 ,s29 ,s30 ,s31 ,s32 ,s33 ,s34 ,s35 ,s36 ; 
__m512d  v0_vec ,v1_vec,v2_vec,v3_vec,v4_vec,v5_vec,v6_vec; 


int m_iter = m / 8;
int m_left = m % 8;


__m512d  tmp_vec;
  g1=_mm512_set1_pd(G[2 * (i+0) + (g+0) * ldg]); 
  s1=_mm512_set1_pd(G[2 * (i+0) + (g+0) * ldg+1]); 


  g2=_mm512_set1_pd(G[2 * (i+0) + (g+1) * ldg]); 
  s2=_mm512_set1_pd(G[2 * (i+0) + (g+1) * ldg+1]); 


  g3=_mm512_set1_pd(G[2 * (i+0) + (g+2) * ldg]); 
  s3=_mm512_set1_pd(G[2 * (i+0) + (g+2) * ldg+1]); 


  g4=_mm512_set1_pd(G[2 * (i+0) + (g+3) * ldg]); 
  s4=_mm512_set1_pd(G[2 * (i+0) + (g+3) * ldg+1]); 


  g5=_mm512_set1_pd(G[2 * (i+0) + (g+4) * ldg]); 
  s5=_mm512_set1_pd(G[2 * (i+0) + (g+4) * ldg+1]); 


  g6=_mm512_set1_pd(G[2 * (i+0) + (g+5) * ldg]); 
  s6=_mm512_set1_pd(G[2 * (i+0) + (g+5) * ldg+1]); 


  g7=_mm512_set1_pd(G[2 * (i+1) + (g+-1) * ldg]); 
  s7=_mm512_set1_pd(G[2 * (i+1) + (g+-1) * ldg+1]); 


  g8=_mm512_set1_pd(G[2 * (i+1) + (g+0) * ldg]); 
  s8=_mm512_set1_pd(G[2 * (i+1) + (g+0) * ldg+1]); 


  g9=_mm512_set1_pd(G[2 * (i+1) + (g+1) * ldg]); 
  s9=_mm512_set1_pd(G[2 * (i+1) + (g+1) * ldg+1]); 


  g10=_mm512_set1_pd(G[2 * (i+1) + (g+2) * ldg]); 
  s10=_mm512_set1_pd(G[2 * (i+1) + (g+2) * ldg+1]); 


  g11=_mm512_set1_pd(G[2 * (i+1) + (g+3) * ldg]); 
  s11=_mm512_set1_pd(G[2 * (i+1) + (g+3) * ldg+1]); 


  g12=_mm512_set1_pd(G[2 * (i+1) + (g+4) * ldg]); 
  s12=_mm512_set1_pd(G[2 * (i+1) + (g+4) * ldg+1]); 


  g13=_mm512_set1_pd(G[2 * (i+2) + (g+-2) * ldg]); 
  s13=_mm512_set1_pd(G[2 * (i+2) + (g+-2) * ldg+1]); 


  g14=_mm512_set1_pd(G[2 * (i+2) + (g+-1) * ldg]); 
  s14=_mm512_set1_pd(G[2 * (i+2) + (g+-1) * ldg+1]); 


  g15=_mm512_set1_pd(G[2 * (i+2) + (g+0) * ldg]); 
  s15=_mm512_set1_pd(G[2 * (i+2) + (g+0) * ldg+1]); 


  g16=_mm512_set1_pd(G[2 * (i+2) + (g+1) * ldg]); 
  s16=_mm512_set1_pd(G[2 * (i+2) + (g+1) * ldg+1]); 


  g17=_mm512_set1_pd(G[2 * (i+2) + (g+2) * ldg]); 
  s17=_mm512_set1_pd(G[2 * (i+2) + (g+2) * ldg+1]); 


  g18=_mm512_set1_pd(G[2 * (i+2) + (g+3) * ldg]); 
  s18=_mm512_set1_pd(G[2 * (i+2) + (g+3) * ldg+1]); 


  g19=_mm512_set1_pd(G[2 * (i+3) + (g+-3) * ldg]); 
  s19=_mm512_set1_pd(G[2 * (i+3) + (g+-3) * ldg+1]); 


  g20=_mm512_set1_pd(G[2 * (i+3) + (g+-2) * ldg]); 
  s20=_mm512_set1_pd(G[2 * (i+3) + (g+-2) * ldg+1]); 


  g21=_mm512_set1_pd(G[2 * (i+3) + (g+-1) * ldg]); 
  s21=_mm512_set1_pd(G[2 * (i+3) + (g+-1) * ldg+1]); 


  g22=_mm512_set1_pd(G[2 * (i+3) + (g+0) * ldg]); 
  s22=_mm512_set1_pd(G[2 * (i+3) + (g+0) * ldg+1]); 


  g23=_mm512_set1_pd(G[2 * (i+3) + (g+1) * ldg]); 
  s23=_mm512_set1_pd(G[2 * (i+3) + (g+1) * ldg+1]); 


  g24=_mm512_set1_pd(G[2 * (i+3) + (g+2) * ldg]); 
  s24=_mm512_set1_pd(G[2 * (i+3) + (g+2) * ldg+1]); 


  g25=_mm512_set1_pd(G[2 * (i+4) + (g+-4) * ldg]); 
  s25=_mm512_set1_pd(G[2 * (i+4) + (g+-4) * ldg+1]); 


  g26=_mm512_set1_pd(G[2 * (i+4) + (g+-3) * ldg]); 
  s26=_mm512_set1_pd(G[2 * (i+4) + (g+-3) * ldg+1]); 


  g27=_mm512_set1_pd(G[2 * (i+4) + (g+-2) * ldg]); 
  s27=_mm512_set1_pd(G[2 * (i+4) + (g+-2) * ldg+1]); 


  g28=_mm512_set1_pd(G[2 * (i+4) + (g+-1) * ldg]); 
  s28=_mm512_set1_pd(G[2 * (i+4) + (g+-1) * ldg+1]); 


  g29=_mm512_set1_pd(G[2 * (i+4) + (g+0) * ldg]); 
  s29=_mm512_set1_pd(G[2 * (i+4) + (g+0) * ldg+1]); 


  g30=_mm512_set1_pd(G[2 * (i+4) + (g+1) * ldg]); 
  s30=_mm512_set1_pd(G[2 * (i+4) + (g+1) * ldg+1]); 


  g31=_mm512_set1_pd(G[2 * (i+5) + (g+-5) * ldg]); 
  s31=_mm512_set1_pd(G[2 * (i+5) + (g+-5) * ldg+1]); 


  g32=_mm512_set1_pd(G[2 * (i+5) + (g+-4) * ldg]); 
  s32=_mm512_set1_pd(G[2 * (i+5) + (g+-4) * ldg+1]); 


  g33=_mm512_set1_pd(G[2 * (i+5) + (g+-3) * ldg]); 
  s33=_mm512_set1_pd(G[2 * (i+5) + (g+-3) * ldg+1]); 


  g34=_mm512_set1_pd(G[2 * (i+5) + (g+-2) * ldg]); 
  s34=_mm512_set1_pd(G[2 * (i+5) + (g+-2) * ldg+1]); 


  g35=_mm512_set1_pd(G[2 * (i+5) + (g+-1) * ldg]); 
  s35=_mm512_set1_pd(G[2 * (i+5) + (g+-1) * ldg+1]); 


  g36=_mm512_set1_pd(G[2 * (i+5) + (g+0) * ldg]); 
  s36=_mm512_set1_pd(G[2 * (i+5) + (g+0) * ldg+1]); 


 for (int j = 0; j < m_iter; j++,v0 += 8,v1 += 8,v2 += 8,v3 += 8,v4 += 8,v5 += 8,v6 += 8,v7 += 8,v8 += 8,v9 += 8,v10 += 8,v11 += 8)
{
  v0_vec = _mm512_loadu_pd(v0);
  v1_vec = _mm512_loadu_pd(v1);
  v2_vec = _mm512_loadu_pd(v2);
  v3_vec = _mm512_loadu_pd(v3);
  v4_vec = _mm512_loadu_pd(v4);
  v5_vec = _mm512_loadu_pd(v5);
  v6_vec = _mm512_loadu_pd(v6);
  tmp_vec = v0_vec;
  v0_vec= _mm512_add_pd(_mm512_mul_pd(g1, tmp_vec), _mm512_mul_pd(s1, v1_vec));
  v1_vec = _mm512_sub_pd(_mm512_mul_pd(g1, v1_vec), _mm512_mul_pd(s1, tmp_vec));


  tmp_vec = v1_vec;
  v1_vec= _mm512_add_pd(_mm512_mul_pd(g2, tmp_vec), _mm512_mul_pd(s2, v2_vec));
  v2_vec = _mm512_sub_pd(_mm512_mul_pd(g2, v2_vec), _mm512_mul_pd(s2, tmp_vec));


  tmp_vec = v2_vec;
  v2_vec= _mm512_add_pd(_mm512_mul_pd(g3, tmp_vec), _mm512_mul_pd(s3, v3_vec));
  v3_vec = _mm512_sub_pd(_mm512_mul_pd(g3, v3_vec), _mm512_mul_pd(s3, tmp_vec));


  tmp_vec = v3_vec;
  v3_vec= _mm512_add_pd(_mm512_mul_pd(g4, tmp_vec), _mm512_mul_pd(s4, v4_vec));
  v4_vec = _mm512_sub_pd(_mm512_mul_pd(g4, v4_vec), _mm512_mul_pd(s4, tmp_vec));


  tmp_vec = v4_vec;
  v4_vec= _mm512_add_pd(_mm512_mul_pd(g5, tmp_vec), _mm512_mul_pd(s5, v5_vec));
  v5_vec = _mm512_sub_pd(_mm512_mul_pd(g5, v5_vec), _mm512_mul_pd(s5, tmp_vec));


  tmp_vec = v5_vec;
  v5_vec= _mm512_add_pd(_mm512_mul_pd(g6, tmp_vec), _mm512_mul_pd(s6, v6_vec));
  v6_vec = _mm512_sub_pd(_mm512_mul_pd(g6, v6_vec), _mm512_mul_pd(s6, tmp_vec));


  _mm512_storeu_pd(v6, v6_vec);
  v6_vec = _mm512_loadu_pd(v7);
  tmp_vec = v6_vec;
  v6_vec= _mm512_add_pd(_mm512_mul_pd(g7, tmp_vec), _mm512_mul_pd(s7, v0_vec));
  v0_vec = _mm512_sub_pd(_mm512_mul_pd(g7, v0_vec), _mm512_mul_pd(s7, tmp_vec));


  tmp_vec = v0_vec;
  v0_vec= _mm512_add_pd(_mm512_mul_pd(g8, tmp_vec), _mm512_mul_pd(s8, v1_vec));
  v1_vec = _mm512_sub_pd(_mm512_mul_pd(g8, v1_vec), _mm512_mul_pd(s8, tmp_vec));


  tmp_vec = v1_vec;
  v1_vec= _mm512_add_pd(_mm512_mul_pd(g9, tmp_vec), _mm512_mul_pd(s9, v2_vec));
  v2_vec = _mm512_sub_pd(_mm512_mul_pd(g9, v2_vec), _mm512_mul_pd(s9, tmp_vec));


  tmp_vec = v2_vec;
  v2_vec= _mm512_add_pd(_mm512_mul_pd(g10, tmp_vec), _mm512_mul_pd(s10, v3_vec));
  v3_vec = _mm512_sub_pd(_mm512_mul_pd(g10, v3_vec), _mm512_mul_pd(s10, tmp_vec));


  tmp_vec = v3_vec;
  v3_vec= _mm512_add_pd(_mm512_mul_pd(g11, tmp_vec), _mm512_mul_pd(s11, v4_vec));
  v4_vec = _mm512_sub_pd(_mm512_mul_pd(g11, v4_vec), _mm512_mul_pd(s11, tmp_vec));


  tmp_vec = v4_vec;
  v4_vec= _mm512_add_pd(_mm512_mul_pd(g12, tmp_vec), _mm512_mul_pd(s12, v5_vec));
  v5_vec = _mm512_sub_pd(_mm512_mul_pd(g12, v5_vec), _mm512_mul_pd(s12, tmp_vec));


  _mm512_storeu_pd(v5, v5_vec);
  v5_vec = _mm512_loadu_pd(v8);
  tmp_vec = v5_vec;
  v5_vec= _mm512_add_pd(_mm512_mul_pd(g13, tmp_vec), _mm512_mul_pd(s13, v6_vec));
  v6_vec = _mm512_sub_pd(_mm512_mul_pd(g13, v6_vec), _mm512_mul_pd(s13, tmp_vec));


  tmp_vec = v6_vec;
  v6_vec= _mm512_add_pd(_mm512_mul_pd(g14, tmp_vec), _mm512_mul_pd(s14, v0_vec));
  v0_vec = _mm512_sub_pd(_mm512_mul_pd(g14, v0_vec), _mm512_mul_pd(s14, tmp_vec));


  tmp_vec = v0_vec;
  v0_vec= _mm512_add_pd(_mm512_mul_pd(g15, tmp_vec), _mm512_mul_pd(s15, v1_vec));
  v1_vec = _mm512_sub_pd(_mm512_mul_pd(g15, v1_vec), _mm512_mul_pd(s15, tmp_vec));


  tmp_vec = v1_vec;
  v1_vec= _mm512_add_pd(_mm512_mul_pd(g16, tmp_vec), _mm512_mul_pd(s16, v2_vec));
  v2_vec = _mm512_sub_pd(_mm512_mul_pd(g16, v2_vec), _mm512_mul_pd(s16, tmp_vec));


  tmp_vec = v2_vec;
  v2_vec= _mm512_add_pd(_mm512_mul_pd(g17, tmp_vec), _mm512_mul_pd(s17, v3_vec));
  v3_vec = _mm512_sub_pd(_mm512_mul_pd(g17, v3_vec), _mm512_mul_pd(s17, tmp_vec));


  tmp_vec = v3_vec;
  v3_vec= _mm512_add_pd(_mm512_mul_pd(g18, tmp_vec), _mm512_mul_pd(s18, v4_vec));
  v4_vec = _mm512_sub_pd(_mm512_mul_pd(g18, v4_vec), _mm512_mul_pd(s18, tmp_vec));


  _mm512_storeu_pd(v4, v4_vec);
  v4_vec = _mm512_loadu_pd(v9);
  tmp_vec = v4_vec;
  v4_vec= _mm512_add_pd(_mm512_mul_pd(g19, tmp_vec), _mm512_mul_pd(s19, v5_vec));
  v5_vec = _mm512_sub_pd(_mm512_mul_pd(g19, v5_vec), _mm512_mul_pd(s19, tmp_vec));


  tmp_vec = v5_vec;
  v5_vec= _mm512_add_pd(_mm512_mul_pd(g20, tmp_vec), _mm512_mul_pd(s20, v6_vec));
  v6_vec = _mm512_sub_pd(_mm512_mul_pd(g20, v6_vec), _mm512_mul_pd(s20, tmp_vec));


  tmp_vec = v6_vec;
  v6_vec= _mm512_add_pd(_mm512_mul_pd(g21, tmp_vec), _mm512_mul_pd(s21, v0_vec));
  v0_vec = _mm512_sub_pd(_mm512_mul_pd(g21, v0_vec), _mm512_mul_pd(s21, tmp_vec));


  tmp_vec = v0_vec;
  v0_vec= _mm512_add_pd(_mm512_mul_pd(g22, tmp_vec), _mm512_mul_pd(s22, v1_vec));
  v1_vec = _mm512_sub_pd(_mm512_mul_pd(g22, v1_vec), _mm512_mul_pd(s22, tmp_vec));


  tmp_vec = v1_vec;
  v1_vec= _mm512_add_pd(_mm512_mul_pd(g23, tmp_vec), _mm512_mul_pd(s23, v2_vec));
  v2_vec = _mm512_sub_pd(_mm512_mul_pd(g23, v2_vec), _mm512_mul_pd(s23, tmp_vec));


  tmp_vec = v2_vec;
  v2_vec= _mm512_add_pd(_mm512_mul_pd(g24, tmp_vec), _mm512_mul_pd(s24, v3_vec));
  v3_vec = _mm512_sub_pd(_mm512_mul_pd(g24, v3_vec), _mm512_mul_pd(s24, tmp_vec));


  _mm512_storeu_pd(v3, v3_vec);
  v3_vec = _mm512_loadu_pd(v10);
  tmp_vec = v3_vec;
  v3_vec= _mm512_add_pd(_mm512_mul_pd(g25, tmp_vec), _mm512_mul_pd(s25, v4_vec));
  v4_vec = _mm512_sub_pd(_mm512_mul_pd(g25, v4_vec), _mm512_mul_pd(s25, tmp_vec));


  tmp_vec = v4_vec;
  v4_vec= _mm512_add_pd(_mm512_mul_pd(g26, tmp_vec), _mm512_mul_pd(s26, v5_vec));
  v5_vec = _mm512_sub_pd(_mm512_mul_pd(g26, v5_vec), _mm512_mul_pd(s26, tmp_vec));


  tmp_vec = v5_vec;
  v5_vec= _mm512_add_pd(_mm512_mul_pd(g27, tmp_vec), _mm512_mul_pd(s27, v6_vec));
  v6_vec = _mm512_sub_pd(_mm512_mul_pd(g27, v6_vec), _mm512_mul_pd(s27, tmp_vec));


  tmp_vec = v6_vec;
  v6_vec= _mm512_add_pd(_mm512_mul_pd(g28, tmp_vec), _mm512_mul_pd(s28, v0_vec));
  v0_vec = _mm512_sub_pd(_mm512_mul_pd(g28, v0_vec), _mm512_mul_pd(s28, tmp_vec));


  tmp_vec = v0_vec;
  v0_vec= _mm512_add_pd(_mm512_mul_pd(g29, tmp_vec), _mm512_mul_pd(s29, v1_vec));
  v1_vec = _mm512_sub_pd(_mm512_mul_pd(g29, v1_vec), _mm512_mul_pd(s29, tmp_vec));


  tmp_vec = v1_vec;
  v1_vec= _mm512_add_pd(_mm512_mul_pd(g30, tmp_vec), _mm512_mul_pd(s30, v2_vec));
  v2_vec = _mm512_sub_pd(_mm512_mul_pd(g30, v2_vec), _mm512_mul_pd(s30, tmp_vec));


  _mm512_storeu_pd(v2, v2_vec);
  v2_vec = _mm512_loadu_pd(v11);
  tmp_vec = v2_vec;
  v2_vec= _mm512_add_pd(_mm512_mul_pd(g31, tmp_vec), _mm512_mul_pd(s31, v3_vec));
  v3_vec = _mm512_sub_pd(_mm512_mul_pd(g31, v3_vec), _mm512_mul_pd(s31, tmp_vec));


  tmp_vec = v3_vec;
  v3_vec= _mm512_add_pd(_mm512_mul_pd(g32, tmp_vec), _mm512_mul_pd(s32, v4_vec));
  v4_vec = _mm512_sub_pd(_mm512_mul_pd(g32, v4_vec), _mm512_mul_pd(s32, tmp_vec));


  tmp_vec = v4_vec;
  v4_vec= _mm512_add_pd(_mm512_mul_pd(g33, tmp_vec), _mm512_mul_pd(s33, v5_vec));
  v5_vec = _mm512_sub_pd(_mm512_mul_pd(g33, v5_vec), _mm512_mul_pd(s33, tmp_vec));


  tmp_vec = v5_vec;
  v5_vec= _mm512_add_pd(_mm512_mul_pd(g34, tmp_vec), _mm512_mul_pd(s34, v6_vec));
  v6_vec = _mm512_sub_pd(_mm512_mul_pd(g34, v6_vec), _mm512_mul_pd(s34, tmp_vec));


  tmp_vec = v6_vec;
  v6_vec= _mm512_add_pd(_mm512_mul_pd(g35, tmp_vec), _mm512_mul_pd(s35, v0_vec));
  v0_vec = _mm512_sub_pd(_mm512_mul_pd(g35, v0_vec), _mm512_mul_pd(s35, tmp_vec));


  tmp_vec = v0_vec;
  v0_vec= _mm512_add_pd(_mm512_mul_pd(g36, tmp_vec), _mm512_mul_pd(s36, v1_vec));
  v1_vec = _mm512_sub_pd(_mm512_mul_pd(g36, v1_vec), _mm512_mul_pd(s36, tmp_vec));


  _mm512_storeu_pd(v11, v2_vec);
  _mm512_storeu_pd(v10, v3_vec);
  _mm512_storeu_pd(v9, v4_vec);
  _mm512_storeu_pd(v8, v5_vec);
  _mm512_storeu_pd(v7, v6_vec);
  _mm512_storeu_pd(v0, v0_vec);
  _mm512_storeu_pd(v1, v1_vec);
}
  if (m_left > 0)
{
 __mmask8 mask = (__mmask8)(255 >> (8 - m_left));


  v0_vec = _mm512_maskz_loadu_pd(mask, v0);
  v1_vec = _mm512_maskz_loadu_pd(mask, v1);
  v2_vec = _mm512_maskz_loadu_pd(mask, v2);
  v3_vec = _mm512_maskz_loadu_pd(mask, v3);
  v4_vec = _mm512_maskz_loadu_pd(mask, v4);
  v5_vec = _mm512_maskz_loadu_pd(mask, v5);
  v6_vec = _mm512_maskz_loadu_pd(mask, v6);


  tmp_vec = v0_vec;
  v0_vec = _mm512_add_pd(_mm512_mul_pd(g1, tmp_vec), _mm512_mul_pd(v1_vec, v0_vec));
  v1_vec = _mm512_sub_pd(_mm512_mul_pd(g1, v1_vec), _mm512_mul_pd(s1, tmp_vec));


  tmp_vec = v1_vec;
  v1_vec = _mm512_add_pd(_mm512_mul_pd(g2, tmp_vec), _mm512_mul_pd(v2_vec, v1_vec));
  v2_vec = _mm512_sub_pd(_mm512_mul_pd(g2, v2_vec), _mm512_mul_pd(s2, tmp_vec));


  tmp_vec = v2_vec;
  v2_vec = _mm512_add_pd(_mm512_mul_pd(g3, tmp_vec), _mm512_mul_pd(v3_vec, v2_vec));
  v3_vec = _mm512_sub_pd(_mm512_mul_pd(g3, v3_vec), _mm512_mul_pd(s3, tmp_vec));


  tmp_vec = v3_vec;
  v3_vec = _mm512_add_pd(_mm512_mul_pd(g4, tmp_vec), _mm512_mul_pd(v4_vec, v3_vec));
  v4_vec = _mm512_sub_pd(_mm512_mul_pd(g4, v4_vec), _mm512_mul_pd(s4, tmp_vec));


  tmp_vec = v4_vec;
  v4_vec = _mm512_add_pd(_mm512_mul_pd(g5, tmp_vec), _mm512_mul_pd(v5_vec, v4_vec));
  v5_vec = _mm512_sub_pd(_mm512_mul_pd(g5, v5_vec), _mm512_mul_pd(s5, tmp_vec));


  tmp_vec = v5_vec;
  v5_vec = _mm512_add_pd(_mm512_mul_pd(g6, tmp_vec), _mm512_mul_pd(v6_vec, v5_vec));
  v6_vec = _mm512_sub_pd(_mm512_mul_pd(g6, v6_vec), _mm512_mul_pd(s6, tmp_vec));


  _mm512_mask_storeu_pd(v6, mask, v6_vec);
  v2_vec = _mm512_maskz_loadu_pd(mask, v7);
  tmp_vec = v6_vec;
  v6_vec = _mm512_add_pd(_mm512_mul_pd(g7, tmp_vec), _mm512_mul_pd(v0_vec, v6_vec));
  v0_vec = _mm512_sub_pd(_mm512_mul_pd(g7, v0_vec), _mm512_mul_pd(s7, tmp_vec));


  tmp_vec = v0_vec;
  v0_vec = _mm512_add_pd(_mm512_mul_pd(g8, tmp_vec), _mm512_mul_pd(v1_vec, v0_vec));
  v1_vec = _mm512_sub_pd(_mm512_mul_pd(g8, v1_vec), _mm512_mul_pd(s8, tmp_vec));


  tmp_vec = v1_vec;
  v1_vec = _mm512_add_pd(_mm512_mul_pd(g9, tmp_vec), _mm512_mul_pd(v2_vec, v1_vec));
  v2_vec = _mm512_sub_pd(_mm512_mul_pd(g9, v2_vec), _mm512_mul_pd(s9, tmp_vec));


  tmp_vec = v2_vec;
  v2_vec = _mm512_add_pd(_mm512_mul_pd(g10, tmp_vec), _mm512_mul_pd(v3_vec, v2_vec));
  v3_vec = _mm512_sub_pd(_mm512_mul_pd(g10, v3_vec), _mm512_mul_pd(s10, tmp_vec));


  tmp_vec = v3_vec;
  v3_vec = _mm512_add_pd(_mm512_mul_pd(g11, tmp_vec), _mm512_mul_pd(v4_vec, v3_vec));
  v4_vec = _mm512_sub_pd(_mm512_mul_pd(g11, v4_vec), _mm512_mul_pd(s11, tmp_vec));


  tmp_vec = v4_vec;
  v4_vec = _mm512_add_pd(_mm512_mul_pd(g12, tmp_vec), _mm512_mul_pd(v5_vec, v4_vec));
  v5_vec = _mm512_sub_pd(_mm512_mul_pd(g12, v5_vec), _mm512_mul_pd(s12, tmp_vec));


  _mm512_mask_storeu_pd(v5, mask, v5_vec);
  v2_vec = _mm512_maskz_loadu_pd(mask, v8);
  tmp_vec = v5_vec;
  v5_vec = _mm512_add_pd(_mm512_mul_pd(g13, tmp_vec), _mm512_mul_pd(v6_vec, v5_vec));
  v6_vec = _mm512_sub_pd(_mm512_mul_pd(g13, v6_vec), _mm512_mul_pd(s13, tmp_vec));


  tmp_vec = v6_vec;
  v6_vec = _mm512_add_pd(_mm512_mul_pd(g14, tmp_vec), _mm512_mul_pd(v0_vec, v6_vec));
  v0_vec = _mm512_sub_pd(_mm512_mul_pd(g14, v0_vec), _mm512_mul_pd(s14, tmp_vec));


  tmp_vec = v0_vec;
  v0_vec = _mm512_add_pd(_mm512_mul_pd(g15, tmp_vec), _mm512_mul_pd(v1_vec, v0_vec));
  v1_vec = _mm512_sub_pd(_mm512_mul_pd(g15, v1_vec), _mm512_mul_pd(s15, tmp_vec));


  tmp_vec = v1_vec;
  v1_vec = _mm512_add_pd(_mm512_mul_pd(g16, tmp_vec), _mm512_mul_pd(v2_vec, v1_vec));
  v2_vec = _mm512_sub_pd(_mm512_mul_pd(g16, v2_vec), _mm512_mul_pd(s16, tmp_vec));


  tmp_vec = v2_vec;
  v2_vec = _mm512_add_pd(_mm512_mul_pd(g17, tmp_vec), _mm512_mul_pd(v3_vec, v2_vec));
  v3_vec = _mm512_sub_pd(_mm512_mul_pd(g17, v3_vec), _mm512_mul_pd(s17, tmp_vec));


  tmp_vec = v3_vec;
  v3_vec = _mm512_add_pd(_mm512_mul_pd(g18, tmp_vec), _mm512_mul_pd(v4_vec, v3_vec));
  v4_vec = _mm512_sub_pd(_mm512_mul_pd(g18, v4_vec), _mm512_mul_pd(s18, tmp_vec));


  _mm512_mask_storeu_pd(v4, mask, v4_vec);
  v2_vec = _mm512_maskz_loadu_pd(mask, v9);
  tmp_vec = v4_vec;
  v4_vec = _mm512_add_pd(_mm512_mul_pd(g19, tmp_vec), _mm512_mul_pd(v5_vec, v4_vec));
  v5_vec = _mm512_sub_pd(_mm512_mul_pd(g19, v5_vec), _mm512_mul_pd(s19, tmp_vec));


  tmp_vec = v5_vec;
  v5_vec = _mm512_add_pd(_mm512_mul_pd(g20, tmp_vec), _mm512_mul_pd(v6_vec, v5_vec));
  v6_vec = _mm512_sub_pd(_mm512_mul_pd(g20, v6_vec), _mm512_mul_pd(s20, tmp_vec));


  tmp_vec = v6_vec;
  v6_vec = _mm512_add_pd(_mm512_mul_pd(g21, tmp_vec), _mm512_mul_pd(v0_vec, v6_vec));
  v0_vec = _mm512_sub_pd(_mm512_mul_pd(g21, v0_vec), _mm512_mul_pd(s21, tmp_vec));


  tmp_vec = v0_vec;
  v0_vec = _mm512_add_pd(_mm512_mul_pd(g22, tmp_vec), _mm512_mul_pd(v1_vec, v0_vec));
  v1_vec = _mm512_sub_pd(_mm512_mul_pd(g22, v1_vec), _mm512_mul_pd(s22, tmp_vec));


  tmp_vec = v1_vec;
  v1_vec = _mm512_add_pd(_mm512_mul_pd(g23, tmp_vec), _mm512_mul_pd(v2_vec, v1_vec));
  v2_vec = _mm512_sub_pd(_mm512_mul_pd(g23, v2_vec), _mm512_mul_pd(s23, tmp_vec));


  tmp_vec = v2_vec;
  v2_vec = _mm512_add_pd(_mm512_mul_pd(g24, tmp_vec), _mm512_mul_pd(v3_vec, v2_vec));
  v3_vec = _mm512_sub_pd(_mm512_mul_pd(g24, v3_vec), _mm512_mul_pd(s24, tmp_vec));


  _mm512_mask_storeu_pd(v3, mask, v3_vec);
  v2_vec = _mm512_maskz_loadu_pd(mask, v10);
  tmp_vec = v3_vec;
  v3_vec = _mm512_add_pd(_mm512_mul_pd(g25, tmp_vec), _mm512_mul_pd(v4_vec, v3_vec));
  v4_vec = _mm512_sub_pd(_mm512_mul_pd(g25, v4_vec), _mm512_mul_pd(s25, tmp_vec));


  tmp_vec = v4_vec;
  v4_vec = _mm512_add_pd(_mm512_mul_pd(g26, tmp_vec), _mm512_mul_pd(v5_vec, v4_vec));
  v5_vec = _mm512_sub_pd(_mm512_mul_pd(g26, v5_vec), _mm512_mul_pd(s26, tmp_vec));


  tmp_vec = v5_vec;
  v5_vec = _mm512_add_pd(_mm512_mul_pd(g27, tmp_vec), _mm512_mul_pd(v6_vec, v5_vec));
  v6_vec = _mm512_sub_pd(_mm512_mul_pd(g27, v6_vec), _mm512_mul_pd(s27, tmp_vec));


  tmp_vec = v6_vec;
  v6_vec = _mm512_add_pd(_mm512_mul_pd(g28, tmp_vec), _mm512_mul_pd(v0_vec, v6_vec));
  v0_vec = _mm512_sub_pd(_mm512_mul_pd(g28, v0_vec), _mm512_mul_pd(s28, tmp_vec));


  tmp_vec = v0_vec;
  v0_vec = _mm512_add_pd(_mm512_mul_pd(g29, tmp_vec), _mm512_mul_pd(v1_vec, v0_vec));
  v1_vec = _mm512_sub_pd(_mm512_mul_pd(g29, v1_vec), _mm512_mul_pd(s29, tmp_vec));


  tmp_vec = v1_vec;
  v1_vec = _mm512_add_pd(_mm512_mul_pd(g30, tmp_vec), _mm512_mul_pd(v2_vec, v1_vec));
  v2_vec = _mm512_sub_pd(_mm512_mul_pd(g30, v2_vec), _mm512_mul_pd(s30, tmp_vec));


  _mm512_mask_storeu_pd(v2, mask, v2_vec);
  v2_vec = _mm512_maskz_loadu_pd(mask, v11);
  tmp_vec = v2_vec;
  v2_vec = _mm512_add_pd(_mm512_mul_pd(g31, tmp_vec), _mm512_mul_pd(v3_vec, v2_vec));
  v3_vec = _mm512_sub_pd(_mm512_mul_pd(g31, v3_vec), _mm512_mul_pd(s31, tmp_vec));


  tmp_vec = v3_vec;
  v3_vec = _mm512_add_pd(_mm512_mul_pd(g32, tmp_vec), _mm512_mul_pd(v4_vec, v3_vec));
  v4_vec = _mm512_sub_pd(_mm512_mul_pd(g32, v4_vec), _mm512_mul_pd(s32, tmp_vec));


  tmp_vec = v4_vec;
  v4_vec = _mm512_add_pd(_mm512_mul_pd(g33, tmp_vec), _mm512_mul_pd(v5_vec, v4_vec));
  v5_vec = _mm512_sub_pd(_mm512_mul_pd(g33, v5_vec), _mm512_mul_pd(s33, tmp_vec));


  tmp_vec = v5_vec;
  v5_vec = _mm512_add_pd(_mm512_mul_pd(g34, tmp_vec), _mm512_mul_pd(v6_vec, v5_vec));
  v6_vec = _mm512_sub_pd(_mm512_mul_pd(g34, v6_vec), _mm512_mul_pd(s34, tmp_vec));


  tmp_vec = v6_vec;
  v6_vec = _mm512_add_pd(_mm512_mul_pd(g35, tmp_vec), _mm512_mul_pd(v0_vec, v6_vec));
  v0_vec = _mm512_sub_pd(_mm512_mul_pd(g35, v0_vec), _mm512_mul_pd(s35, tmp_vec));


  tmp_vec = v0_vec;
  v0_vec = _mm512_add_pd(_mm512_mul_pd(g36, tmp_vec), _mm512_mul_pd(v1_vec, v0_vec));
  v1_vec = _mm512_sub_pd(_mm512_mul_pd(g36, v1_vec), _mm512_mul_pd(s36, tmp_vec));


  _mm512_mask_storeu_pd(v11, mask, v2_vec);
  _mm512_mask_storeu_pd(v10, mask, v3_vec);
  _mm512_mask_storeu_pd(v9, mask, v4_vec);
  _mm512_mask_storeu_pd(v8, mask, v5_vec);
  _mm512_mask_storeu_pd(v7, mask, v6_vec);
  _mm512_mask_storeu_pd(v0, mask, v0_vec);
  _mm512_mask_storeu_pd(v1, mask, v1_vec);
}
}
