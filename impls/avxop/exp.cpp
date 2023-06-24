#include "impls/avxop/avxop_impl.h"

namespace impl {
namespace avxop {
__m256 exp256_ps(__m256 x) {
  __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
  __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);

  __m256 cephes_LOG2EF = _mm256_set1_ps(1.44269504088896341f);
  __m256 inv_LOG2EF = _mm256_set1_ps(0.693147180559945f);

  __m256 cephes_exp_p0 = _mm256_set1_ps(1.9875691500E-4);
  __m256 cephes_exp_p1 = _mm256_set1_ps(1.3981999507E-3);
  __m256 cephes_exp_p2 = _mm256_set1_ps(8.3334519073E-3);
  __m256 cephes_exp_p3 = _mm256_set1_ps(4.1665795894E-2);
  __m256 cephes_exp_p4 = _mm256_set1_ps(1.6666665459E-1);
  __m256 cephes_exp_p5 = _mm256_set1_ps(5.0000001201E-1);
  __m256 fx;
  __m256i imm0;
  __m256 one = _mm256_set1_ps(1.0f);

  x = _mm256_min_ps(x, exp_hi);
  x = _mm256_max_ps(x, exp_lo);

  /* express exp(x) as exp(g + n*log(2)) */
  fx = _mm256_mul_ps(x, cephes_LOG2EF);
  fx = _mm256_round_ps(fx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 z = _mm256_mul_ps(fx, inv_LOG2EF);
  x = _mm256_sub_ps(x, z);
  z = _mm256_mul_ps(x, x);

  __m256 y = cephes_exp_p0;
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, cephes_exp_p1);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, cephes_exp_p2);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, cephes_exp_p3);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, cephes_exp_p4);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, cephes_exp_p5);
  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, x);
  y = _mm256_add_ps(y, one);

  /* build 2^n */
  imm0 = _mm256_cvttps_epi32(fx);
  imm0 = _mm256_add_epi32(imm0, _mm256_set1_epi32(0x7f));
  imm0 = _mm256_slli_epi32(imm0, 23);
  __m256 pow2n = _mm256_castsi256_ps(imm0);
  y = _mm256_mul_ps(y, pow2n);
  return y;
}

__m512 exp512_ps(__m512 x) {
  __m512 exp_hi = _mm512_set1_ps(88.3762626647949f);
  __m512 exp_lo = _mm512_set1_ps(-88.3762626647949f);

  __m512 cephes_LOG2EF = _mm512_set1_ps(1.44269504088896341f);
  __m512 inv_LOG2EF = _mm512_set1_ps(0.693147180559945f);

  __m512 cephes_exp_p0 = _mm512_set1_ps(1.9875691500E-4);
  __m512 cephes_exp_p1 = _mm512_set1_ps(1.3981999507E-3);
  __m512 cephes_exp_p2 = _mm512_set1_ps(8.3334519073E-3);
  __m512 cephes_exp_p3 = _mm512_set1_ps(4.1665795894E-2);
  __m512 cephes_exp_p4 = _mm512_set1_ps(1.6666665459E-1);
  __m512 cephes_exp_p5 = _mm512_set1_ps(5.0000001201E-1);
  __m512 fx;
  __m512i imm0;
  __m512 one = _mm512_set1_ps(1.0f);

  x = _mm512_min_ps(x, exp_hi);
  x = _mm512_max_ps(x, exp_lo);

  /* express exp(x) as exp(g + n*log(2)) */
  fx = _mm512_mul_ps(x, cephes_LOG2EF);
  fx = _mm512_roundscale_ps(fx, _MM_FROUND_TO_NEAREST_INT);
  __m512 z = _mm512_mul_ps(fx, inv_LOG2EF);
  x = _mm512_sub_ps(x, z);
  z = _mm512_mul_ps(x, x);

  __m512 y = cephes_exp_p0;
  y = _mm512_mul_ps(y, x);
  y = _mm512_add_ps(y, cephes_exp_p1);
  y = _mm512_mul_ps(y, x);
  y = _mm512_add_ps(y, cephes_exp_p2);
  y = _mm512_mul_ps(y, x);
  y = _mm512_add_ps(y, cephes_exp_p3);
  y = _mm512_mul_ps(y, x);
  y = _mm512_add_ps(y, cephes_exp_p4);
  y = _mm512_mul_ps(y, x);
  y = _mm512_add_ps(y, cephes_exp_p5);
  y = _mm512_mul_ps(y, z);
  y = _mm512_add_ps(y, x);
  y = _mm512_add_ps(y, one);

  /* build 2^n */
  imm0 = _mm512_cvttps_epi32(fx);
  imm0 = _mm512_add_epi32(imm0, _mm512_set1_epi32(0x7f));
  imm0 = _mm512_slli_epi32(imm0, 23);
  __m512 pow2n = _mm512_castsi512_ps(imm0);
  y = _mm512_mul_ps(y, pow2n);
  return y;
}
} // namespace avxop
} // namespace impl