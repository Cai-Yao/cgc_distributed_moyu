#ifndef GCN_IMPLS_AVXOP_AVXOP_IMPL_H_
#define GCN_IMPLS_AVXOP_AVXOP_IMPL_H_

#include <immintrin.h>

namespace impl {
namespace avxop {
__m256 exp256_ps(__m256 x);
__m512 exp512_ps(__m512 x);

} // namespace avxop
} // namespace impl

#endif