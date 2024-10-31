// function.h
#ifndef APPLY_REV_AVX_H
#define APPLY_REV_AVX_H

void apply_rev_avx();
void apply_rev_avx_mv();
void apply_rev_avx512_mv();
void apply_rev_avx_mv_seq();
void apply_rev_avx512_mv_seq();
void apply_rev_avx_mv_seq_avx256();
void apply_rev_avx_mv_seq_fma();

#endif
