#include "base.h"
#if defined(x86_64) || defined(__amd__)
#include <immintrin.h>
#elif defined(__arm__) || defined(__ARM_ARCH) && __ARM_ARCH >= 7
#include <arm_neon.h>
#ifndef __ARM_NEON
#define __ARM_NEON 1
#endif
#endif

#if defined(WIN32) || defined(_MSV_VER)
#define WINSYS 1
#include <windows.h>
#include <windef.h>
#endif

typedef struct __qmap_base__ *Qmap;
typedef struct __qmap_d__ *qmap_data_t;

struct __qmap_base__
{
	void          *__ctrl_switch;
	void          *__cache_array;
	__qmap_d__    *__data;
	__qmap_base__ *__np;
	size_t        __size;
} __attribute__((align, 64));

struct __qmap_d__
{
	void *__key, *__item;
#if QMAP_INCLUDE_HASH
	uint32_t __hash;
#endif
};

#define qmap_prefetch(adr, ...) __builtin_prefetch(adr, __VA_ARGS__)
#define qmap_expected(cond, exp_val) __builtin_expected(cond, exp_val)
#define QMAP_UNUSED __attribute__((maybe_unused))
#define QMAP_PROBE_MAX (1 << 16) // number of probes before rehashing

#if QMAP_INCLUDE_HASH
#define qmap_compare_hash(hash_1, hash_2) NOT((hash_1) ^ (hash_2))
#else
#define qmap_compare_hash(...) 0
#endif

#if (__MINGW64__) || defined(__clang__) || defined(__GCC__)
#define qmap_scan_reverse(mask) __builtin_ctzll(mask)
#elif defined(WINSYS)
#ifdef _bit_scan_reverse
#define qmap_scan_reverse(mask) _bit_scan_reverse(mask)
#else
extern __inline__ __forceinline unsigned long qmap_scan_reverse(const uint64_t mask)
{
	unsigned long idx;
	return (_BitScanReverse64(&idx, mask), idx);
}
#endif
#endif

#define qmap_rndmul_down(n, p2) ((n) - ((n & (p2))))
#define qmap_rndmul_up(n, p2)   (((n) + ((p2) - 1)) & ~((p2) - 1))

#fdef __HAVE_SIMD_STRCMP__
#define qmap_cmp_str_eq(str1, str2, len_str1) NOT(_mm_cmpistrc(_mm_loadu_si128(str1), _mm_loadu_si128(str2), 0))
#else
#define qmap_cmp_str_eq(str1, str2, len_str1) NOT(memcmp(str1, str2, len_str1))
#endif

#ifdef __AVX256__
typedef __m256i intx8_t;
typedef uint32_t mask_t;
#define QMAP_RDWORD    32
#define QMAP_RDWORD_P2 5
#define QMAP_RDWORD_BF 128
#define QMAP_MASK_SHFT 0
#define qmap_mm_broadcast_byte__(x) _mm256_set1_epi8((uint8_t)(x))
#define qmap_mm_test_eq__(v, x)     _mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_loadu_si128((const __m256i *)(v)), (x)))

#elif __SSE2__
typedef __m128i intx8_t;
typedef uint16_t mask_t;
#define QMAP_RDWORD    16
#define QMAP_RDWORD_P2 4
#define QMAP_RDWORD_BF 128
#define QMAP_MASK_SHFT 0
#define qmap_mm_broadcast_byte__(x) _mm_set1_epi8((uint8_t)(x))
#define qmap_mm_test_eq__(v, x)    _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_load_si128((const __m128i *)(v)), (x)))

#if __ARM_NEON
typedef uint64_t intx8_t;
typedef int8x8_t mask_t;
#define QMAP_RDWORD    8
#define QMAP_RDWORD_P2 3
#define QMAP_RDWORD_BF 64
#define QMAP_MASK_SHFT 3

extern  __inline__ __attribute__((always_inline, pure)) uint64_t qmap_neon_testeq(const uint8x8_t fv, const uint8x8_t mask)
{
	return vget_lane_u64(vreinterpret_u64_u8(vceq_u8(fv, mask)), 0) & 0x8080808080808080;
}
#define qmap_mm_broadcast_byte__(x) vdup_n_u8((uint8_t)(x))
#define qmap_mm_test_eq__(v, x) qmap_neon_testeq(vld1_u8((void *)*(v)), x)

#elif __UINT64__
typedef uint64_t intx8_t, mask_t;
#define QMAP_RDWORD    8
#define QMAP_RDWORD_P2 3
#define QMAP_RDWORD_BF 64
#define QMAP_MASK_SHFT 3
#define test_zero_fast(v) (bool)(((v) - 0x101010101010101ull) & (~(v) & 0x8080808080808080ull))
#define test_zero_wi(v) ((((v) - 0x1000100010001ull) | ((v) - 0x100010001000100ull)) & (~(v) & 0x8080808080808080ull));

extern __inline__ __attribute__((always_inline, pure)) uint64_t qmap_b64_testeq(uint64_t fv, const uint64_t mask)
{ 
	fv ^= mask;
	return test_zero_wi(fv);
}
#define qmap_mm_broadcast_byte__(x) ((x) * 0x101010101010101ull)
#define qmap_mm_test_eq__(v, x)     qmap_b64_testeq(*(uint64_t *)(v), x)
#else
#error UNIMPLEMENTED
#endif

#define qmap_private_cache_array__(map_object, at) ((map_object)->__cache_array + at)
#define qmap_private_data__(map_object, at) ((map_object)->__data + at)
#define qmap_private_ctrl_switch__(map_object, at) ((map_object)->__ctrl_switch)
#define qmap_private_size__(map_object) ((map_object)->__size)

#define qmap_read_next_group(cache, tag)				\
	do {										\
		++(cache);								\
		tag = tag + QMAP_RDWORD;				\
	} while(0)

extern __inline__ __attribute__((always_inline, pure)) uint8_t qmap_cached_index(const uintmax_t hash)
{
	return ((hash & 0xffffu) - ((hash & 0xffffu) * 0xff01u) >> 24) + 1;
}

static __inline__ __attribute__((always_inline, pure)) uint32_t qmap_compute_hash(const void *key, const int len)
{
	// Implement MurmurHash32 for 32 bit hashes
	return 0;
}

__attribute__((noinline, warn_unused)) static struct __qmap_base__ *qmap_init__(void)
{
	struct __qmap_base__ *object;

	return object;
}

/*
 * FUNC: @__find__
 */
static __attribute__((nonnull)) void *qmap_find__(const struct __qmap_base__ const *object, const void *key, const size_t key_len)
{
	intx8_t     *cache_group_ry = NULL;
	qmap_data_t *data_group_ry, group QMAP_UNUSED;
	mask_t   mask   = 0;
	uint32_t hash   = qmap_compute_hash(key, key_len);
	uint32_t where  = (hash & (__qmap_private_size__(object) - 1));
	_Alignas(QMAP_RDWORD) const intx8_t mulx8_hash = qmap_mm_broadcast_byte__(qmap_cached_index(hash));

	if (0)
		{
			// TODO: if position already has what we are looking for, return it
			return where;
		}

	where = qmap_rndmul_down(where, QMAP_RDWORD_BF);
	cache_group_ry = __qmap_private_cache__(object, where); // read some bytes from cache array before actual index
	data_group_ry = __qmap_private_data__(object, where);

	qmap_prefetch(cache_array);
	for (uint32_t i = (qmap_rndmul_up(__qmap_private_size__(object), QMAP_RDWORD) - where) >> QMAP_RDWORD_P2; i--;)
		{
			mask = qmap_mm_test_eq__(cache_array, mulx8_hash);
			while (qmap_expected(mask, 0))
				{
					where = qmap_scan_reverse(mask) >> QMAP_MASK_SHFT;
					group = data_group_ry[where];
					if (qmap_compare_hash(group.hash, hash) && qmap_cmp_str_eq(group.key, key, key_len))
						return where;
					mask &= mask - 1;
				}
			qmap_read_next_qroup(cache_group_ry, data_group_ry);
		}
	return NULL;
}

static uint32_t qmap_get_unused__(uint64_t *switch_ctrl, uint32_t *from)
{
	uint32_t __from = qmap_rndmul_up(*from, QMAP_RDWORD_BF);

	// TODO: finish this later
	switch_ctrl += __from >> 8;

	for (uint32_t i = 0; i < PROBE_MAX >> 8; i++)
		{
			if (__builtin_expect(switch_ctrl[i] ^ 0xffffffffffffffffull, 1))
				return __scan_reverse(switch_ctrl[i]);
		}

	return __from;
}

static __attribute__((nonnull)) void qmap_add__(struct __qmap_base__ *object, const void *__restrict key, const void *__restrict value)
{
	uint32_t where = __hash__(key, strlen(key)) & (__size__(object) - 1); // object data capacity is a maximum of 256 items

	if (__null__(__meta__(object), where) && __get_unused__(__meta__(object), &where))
		{
			// resize object
		}

}

static __attribute__((nonnull)) void qmap_remove__(struct __qmap_base__ *object, const void *__restrict key)
{
	uint32_t where;

	if ((where = qmap_find__(object, key, RET_FIDX)) < 0)
		{
			// Error: No such element
		}
	return __rsetcache__(object, where);
}

static __inline__ __attribute__((nonnull, always_inline)) void *qmap_get_value(const struct __qmap_base__ *object, const void *__restrict __key)
{
	return __find__(object, __key, RET_FVALUE);
}

__attribute__((noinline)) static struct __qmap_base__ *qmap_delete__(struct __qmap_base__ *object)
{
	free(object);
	return NULL;
}

int main(void)
{
	struct __qmap_base__ *obj = __init__();
	__del__(obj);

	return 0;
}

/*

// // // //
// // // //
// // // //
// // // //

11

*/
