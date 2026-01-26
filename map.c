#include "base.h"
#define __amd__
#define __SSE2__ 1

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

typedef struct __qmap_base__ __qmap_base__, *Qmap;
typedef struct __qmap_d__ __qmap_d__, *qmap_data_t;

struct __qmap_base__
{
	void *__ctrl_switch;
	uint8_t *__cache_array;
	__qmap_d__ *__data;
	__qmap_base__ *__np;
	size_t __capacity;
	size_t __size;
	size_t __ldf;
	uint16_t __rehash_size;
} __attribute__((aligned(64)));

struct __qmap_d__
{
	void *__key, *__value;
#if QMAP_INCLUDE_HASH
	uint32_t __hash;
#endif
};

#define qmap_prefetch(adr, ...) __builtin_prefetch(adr, __VA_ARGS__)
#define qmap_expect(cond, exp_val) __builtin_expect(cond, exp_val)
#define QMAP_UNUSED __attribute__((unused))
#define QMAP_DEFAULT_SIZE 32
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
static uint64_t qmap_scan_reverse(const uint64_t);
extern __inline__ __forceinline uint64_t qmap_scan_reverse(const uint64_t mask)
{
	unsigned long idx;
	return (_BitScanReverse64(&idx, mask), idx);
}
#endif
#endif

#define qmap_rndmul_down(n, p2) ((n) - ((n & (p2))))
#define qmap_rndmul_up(n, p2) (((n) + ((p2) - 1)) & ~((p2) - 1))

#if defined(__HAVE_SIMD_STRCMP__)
#define qmap_cmp_str_eq(str1, str2, len_str1) NOT(_mm_cmpistrc(_mm_loadu_si128(str1), _mm_loadu_si128(str2), 0))
#else
#define qmap_cmp_str_eq(str1, str2, len_str1) NOT(memcmp(str1, str2, len_str1))
#endif

#ifdef __AVX256__
typedef __m256i intx8_t;
typedef uint32_t mask_t;
#define QMAP_RDWORD 32
#define QMAP_RDWORD_P2 5
#define QMAP_RDWORD_BF 128
#define QMAP_MASK_SHFT 0
#define qmap_mm_broadcast_byte__(x) _mm256_set1_epi8((uint8_t)(x))
#define qmap_mm_set_zero__() _mm256_setzero_si256()
#define qmap_mm_test_eq__(v, x) _mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_loadu_si128((const __m256i *)(v)), (x)))

#elif defined(__SSE2__)
typedef __m128i intx8_t;
typedef uint16_t mask_t;
#define QMAP_RDWORD 16
#define QMAP_RDWORD_P2 4
#define QMAP_RDWORD_BF 128
#define QMAP_MASK_SHFT 0
#define qmap_mm_broadcast_byte__(x) _mm_set1_epi8((uint8_t)(x))
#define qmap_mm_set_zero__() _mm_setzero_si128();
#define qmap_mm_test_eq__(v, x) _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_load_si128((const __m128i *)(v)), (x)))

#elif defined(__ARM_NEON)
typedef uint64_t intx8_t;
typedef int8x8_t mask_t;
static uint64_t qmap_neon_testeq(const uint8x8_t, const uint8x8_t);
#define QMAP_RDWORD 8
#define QMAP_RDWORD_P2 3
#define QMAP_RDWORD_BF 64
#define QMAP_MASK_SHFT 3

extern __inline__ __attribute__((always_inline, pure)) uint64_t qmap_neon_testeq(const uint8x8_t fv, const uint8x8_t mask)
{
	return vget_lane_u64(vreinterpret_u64_u8(vceq_u8(fv, mask)), 0) & 0x8080808080808080;
}
#define qmap_mm_broadcast_byte__(x) vdup_n_u8((uint8_t)(x))
#define qmap_mm_set_zero__() qmap_mm_broadcast_byte__(0)
#define qmap_mm_test_eq__(v, x) qmap_neon_testeq(vld1_u8((void *)*(v)), x)

#elif defined(__UINT64__)
typedef uint64_t intx8_t, mask_t;
static uint64_t qmap_b64_testeq(uint64_t, const uint64_t);
#define QMAP_RDWORD 8
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
#define qmap_mm_test_eq__(v, x) qmap_b64_testeq(*(uint64_t *)(v), x)
#define qmap_mm_set_zero__() 0
#else
#error UNIMPLEMENTED
#endif

#define qmap_private_cache_array__(map_object, at) (&((map_object)->__cache_array))[at]
#define qmap_private_data__(map_object, at) (&((map_object)->__data))[at]
#define qmap_private_ctrl_switch__(map_object, at) (&((map_object)->__ctrl_switch))[at]
#define qmap_private_capacity__(map_object) ((map_object)->__capacity)
#define qmap_private_size__(map_object) ((map_object)->__size)
#define qmap_private_load_factor__(map_object) ((map_object)->__ldf)
extern __inline__ __attribute__((always_inline)) void qmap_private_set_structure__(__qmap_base__ *map, const size_t i, const void *k, const void *v, const uint32_t h)
{
	qmap_data_t _map_d = qmap_private_data__(map, i);

	_map_d->__key = k, _map_d->__value = v;
#ifdef QMAP_INCLUDE_HASH
	_map_d->__hash = h;
#endif
	qmap_private_cache_array__(map, i)[0] = qmap_cached_index(h);
	qmap_private_size__(map) += 1;
}

#define qmap_read_next_group(cache_array, data_array) \
	do                                                \
	{                                                 \
		++(cache_array);                              \
		data_array += QMAP_RDWORD;                    \
	} while (0)

#define qmap_read_next_cache_group(cache_array) \
	do                                          \
	{                                           \
		++(cache_array);                        \
	} while (0)

extern __inline__ __attribute__((always_inline, pure)) uint8_t qmap_cached_index(const uintmax_t hash)
{
	return (hash & 0xffffu) - (((hash & 0xffffu) * 0xff01u) >> 24) + 1;
}

static __inline__ __attribute__((always_inline, pure)) uint32_t qmap_compute_hash(const uint8_t *key, const int len)
{
	// Implement MurmurHash32 for 32 bit hashes

	unsigned long hash = 5381u, c;
	while ((c = *key++))
	{
		hash = ((hash << 5) + hash) + c;
	}

	return hash;
}

__attribute__((noinline, warn_unused)) static struct __qmap_base__ *qmap_init__(void)
{
	struct __qmap_base__ *object = qmap_calloc(1, sizeof(__qmap_base__));
	return object;
}

__attribute__((nonnull)) void qmap_reserve__(__qmap_base__ *object, size_t size)
{
	size_t capsize;

	if (size == 0)
		return;
	if (qmap_private_capacity__(object) != 0)
	{
		// TODO: if object is nonempty and reserve required a size > capacity, resize and rehash object else do nothing
		return;
	}
	capsize = qmap_rndmul_up(size, QMAP_RDWORD);
	qmap_private_cache_array__(object, 0) = qmap_calloc(1, capsize + (capsize >> 6));
	qmap_private_ctrl_switch__(object, 0) = (uint8_t *)qmap_private_cache_array__(object, capsize);
	qmap_private_data__(object, 0) = qmap_calloc(capsize, sizeof(__qmap_d__));
	qmap_private_capacity__(object) = capsize;
	qmap_private_load_factor__(object) = (capsize * 0.875) + 0.5;
}

/*
 * FUNC: @__find__
 */
static __attribute__((nonnull)) size_t qmap_find__(const struct __qmap_base__ const *object, const void *key, const size_t key_len)
{
	intx8_t *cache_group_ry = NULL;
	qmap_data_t *data_group_ry, group QMAP_UNUSED;
	mask_t mask = 0;
	uint32_t hash = qmap_compute_hash(key, key_len);
	uint32_t where = (hash & (qmap_private_capacity__(object) - 1));
	_Alignas(QMAP_RDWORD) const intx8_t mulx8_hash = qmap_mm_broadcast_byte__(qmap_cached_index(hash));

	if (0)
	{
		// TODO: if position already has what we are looking for, return it
		return where;
	}

	where = qmap_rndmul_down(where, QMAP_RDWORD_BF);
	cache_group_ry = qmap_private_cache__(object, where); // read some bytes from cache array before actual index
	data_group_ry = qmap_private_data__(object, where);

	qmap_prefetch(cache_group_ry, 0, 3);
	for (uint32_t i = (qmap_rndmul_up(qmap_private_size__(object), QMAP_RDWORD) - where) >> QMAP_RDWORD_P2; i--;)
	{
		mask = qmap_mm_test_eq__(cache_group_ry, mulx8_hash);
		while (qmap_expect(mask, 0))
		{
			where = qmap_scan_reverse(mask) >> QMAP_MASK_SHFT;
			group = data_group_ry[where];
			if (qmap_compare_hash(group.hash, hash) && qmap_cmp_str_eq(group->__key, key, key_len))
				return where;
			mask &= mask - 1;
		}
		qmap_read_next_qroup(cache_group_ry, data_group_ry);
	}
	return NULL;
}

#ifdef QMAP_USMALL_SPACE
static int qmap_get_unused__(uint8_t *cache_array, size_t size, uint32_t *from)
{
	size_t index = qmap_rndmul_down(*from, QMAP_RDWORD_BF);
	mask_t mask = 0;
	intx8_t *cache_group_ry = cache_array + index;
	_Alignas(QMAP_RDWORD) const intx8_t mulx8_zero = qmap_mm_set_zero__();

	for (size_t i = 0, e = (qmap_rndmul_up(size, QMAP_RDWORD) - index) >> 6; i < e; i++)
	{
		mask = qmap_mm_test_eq__(cache_group_ry, mulx8_zero);
		if (qmap_expect(mask, 1))
			return (*from = (i << QMAP_RDWORD_P2) | qmap_scan_reverse(mask));
		qmap_read_next_cache_group(cache_array);
	}
	return -1;
}
#define qmap_get_unused(map_object, at) qmap_get_unused__(qmap_private_cache_array__(map_object), qmap_private_size__(map_object), at)
#define qmap_ctrl_is_empty(map_object) (qmap_private_cache_array__(map_object, at)[0] == 0)
#else
static int qmap_get_unused__(uint64_t *ctrl_switch, size_t size, uint32_t *from)
{
	size_t index = qmap_rndmul_down(*from, QMAP_RDWORD_BF);

	ctrl_switch += index >> 6;
	for (size_t i = 0, e = (qmap_rndmul_up(size, QMAP_RDWORD) - index) >> 6; i < e; i++)
		if (qmap_expect(*ctrl_switch ^ 0xffffffffffffffffull, 1))
		{
			const uint64_t qsr = qmap_scan_reverse(*ctrl_switch);
			*ctrl_switch |= (1ull << qsr);
			return (*from = (i << 8) | qsr);
		}

	return -1;
}
#define qmap_get_unused(map_object, at) qmap_get_unused__(qmap_private_ctrl_switch__(map_object, 0), qmap_private_size__(map_object), at)
#define qmap_ctrl_is_empty(map_object, at) qmap_private_ctrl_switch__(map_object, at >> 6) & (1ull << ((at) & 63u))
#endif

static __attribute__((nonnull)) void qmap_add__(struct __qmap_base__ *object, const void *__restrict key, const void *__restrict value)
{
	void *ctrl_switch;
	uint32_t where QMAP_UNUSED, hash QMAP_UNUSED;

	if (qmap_empty_object(object))
		qmap_reserve(object, QMAP_DEFAULT_SIZE);

	if (qmap_expect(qmap_private_size__(object) > qmap_private_ldf_size__(object), 0))
		goto rehash;

	hash = qmap_compute_hash(key, strlen(key));
	where = hash & (qmap_private_size__(object) - 1);

	if (!qmap_ctrl_is_empty(object, where) && qmap_get_unused(object, &where) < 0)
	{
	rehash:
		// resize object
		return;
	}
	qmap_private_set_structure__(object, where, key, value, hash);
}

static __attribute__((nonnull)) void qmap_remove__(struct __qmap_base__ *object, const void *__restrict key)
{
	uint32_t where;

	if ((where = qmap_find__(object, key, strlen(key))) < 0)
	{
		// Error: No such element
	}
	return __rsetcache__(object, where);
}

static __inline__ __attribute__((nonnull, always_inline)) void *qmap_get_value(const struct __qmap_base__ *object, const void *__restrict __key)
{
	return __find__(object, __key, 0);
}

__attribute__((noinline)) static struct __qmap_base__ *qmap_delete__(struct __qmap_base__ *object)
{
	return NULL;
}
