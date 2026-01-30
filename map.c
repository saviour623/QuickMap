#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stddef.h>
#include <error.h>
#if defined(__x86_64__) || defined(__i386__) || defined(__amd__)
#include <immintrin.h>
#elif defined(__aarch64__) || defined(__arm__) || defined(__ARM_ARCH) && __ARM_ARCH >= 7
#include <arm_neon.h>
#ifndef __ARM_NEON
#define __ARM_NEON 1
#endif
#endif
#if defined(_WIN32) || defined(_MSV_VER)
#define WINSYS 1
#include <windows.h>
#include <windef.h>
#endif

typedef struct __qmap_base__ __qmap_base__, *qmap_unordered_map_t;
typedef struct __qmap_kwvh__ __qmap_kwvh__, *qmap_kwvhpair_t;
static uint8_t qmap_cached_index(const uint16_t);

struct __qmap_base__
{
	void        *__ctrl_switch;
	uint8_t     *__cache;
	qmap_kwvhpair_t __data;
	size_t   __capacity;
	size_t   __size;
	size_t   __load_factor;
	uint16_t __rehash_size;
} __attribute__((aligned(64)));

struct __qmap_kwvh__
{
	void *__key, *__value;
#if QMAP_INCLUDE_HASH
	uint32_t __hash;
#endif
};

typedef struct {
	uintmax_t (*__hash_func)(void *key, size_t key_len, uint16_t seed);
	void *    (*__new)(size_t n, size_t size);
	void      (*__delete)(void *ptr);
	void      (*__kwvh_delete)(void *ptr);
	uint16_t   *__seed;
} qmap_policy_t;

#define qmap_prefetch(adr, ...) __builtin_prefetch(adr, __VA_ARGS__)
#define qmap_expect(cond, exp_val) __builtin_expect(cond, exp_val)
#define QMAP_UNUSED __attribute__((unused))
#define QMAP_DEFAULT_SIZE 32
#define QMAP_NULL 0

#if QMAP_INCLUDE_HASH
#define qmap_compare_hash(_1, _2) !((_1) ^ (_2))
#else
#define qmap_compare_hash(...) (1)
#endif

#if defined(__MINGW64__) || defined(__clang__) || defined(__GCC__)
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
#define qmap_cmp_str_eq(str1, str2, len_str1) !(_mm_cmpistrc(_mm_loadu_si128(str1), _mm_loadu_si128(str2), 0))
#else
#define qmap_cmp_str_eq(str1, str2, len_str1) !(memcmp(str1, str2, len_str1))
#endif

#ifdef __AVX2__
typedef __m256i intx8_t;
typedef uint32_t mask_t;
#define QMAP_RDWORD 32
#define QMAP_RDWORD_P2 5
#define QMAP_MASK_SHFT 0
#define qmap_mm_broadcast_byte__(x) _mm256_set1_epi8((uint8_t)(x))
#define qmap_mm_set_zero__() _mm256_setzero_si256()
#define qmap_mm_test_eq__(v, x) _mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_loadu_si128((const __m256i *)(v)), (x)))

#elif defined(__SSE2__)
typedef __m128i intx8_t;
typedef uint16_t mask_t;
#define QMAP_RDWORD 16
#define QMAP_RDWORD_P2 4
#define QMAP_MASK_SHFT 0
#define qmap_mm_broadcast_byte__(x) _mm_set1_epi8((uint8_t)(x))
#define qmap_mm_set_zero__() _mm_setzero_si128();
#define qmap_mm_test_eq__(v, x) _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_load_si128((const __m128i *)(v)), (x)))

#elif defined(__ARM_NEON)
typedef uint8x8_t intx8_t;
typedef uint64_t  mask_t;
#define QMAP_RDWORD 8
#define QMAP_RDWORD_P2 3
#define QMAP_MASK_SHFT 3
static uint64_t qmap_neon_testeq(const uint8x8_t, const uint8x8_t);
extern __inline__ __attribute__((always_inline, pure)) uint64_t qmap_neon_testeq(const uint8x8_t fv, const uint8x8_t mask)
{
	return vget_lane_u64(vreinterpret_u64_u8(vceq_u8(fv, mask)), 0) & 0x8080808080808080;
}
#define qmap_mm_broadcast_byte__(x) vdup_n_u8((uint8_t)(x))
#define qmap_mm_set_zero__() qmap_mm_broadcast_byte__(0)
#define qmap_mm_test_eq__(v, x) qmap_neon_testeq(vld1_u8(v), x)

#elif defined(__UINT64__)
typedef uint64_t intx8_t, mask_t;
#define QMAP_RDWORD 8
#define QMAP_RDWORD_P2 3
#define QMAP_MASK_SHFT 3
#define test_zero_fast(v) (bool)(((v) - 0x101010101010101ull) & (~(v) & 0x8080808080808080ull))
#define test_zero_wi(v) ((((v) - 0x1000100010001ull) | ((v) - 0x100010001000100ull)) & (~(v) & 0x8080808080808080ull));
static uint64_t qmap_b64_testeq(uint64_t, const uint64_t);
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

#define qmap__cache__(map_object, at) (&((map_object)->__cache))[at]
#define qmap__data__(map_object, at)        (&((map_object)->__data))[at]
#define qmap__ctrl_switch__(map_object, at) (&((map_object)->__ctrl_switch))[at]
#define qmap__capacity__(map_object)        ((map_object)->__capacity)
#define qmap__size__(map_object)            ((map_object)->__size)
#define qmap__load_factor__(map_object)     ((map_object)->__load_factor)
static __inline__ __attribute__((always_inline)) void qmap__set_structure__(__qmap_base__ *map, size_t i, void *k, void *v, uint32_t h)
{
	qmap_kwvhpair_t _map_d = qmap__data__(map, i);

	_map_d->__key = k, _map_d->__value = v;
#ifdef QMAP_INCLUDE_HASH
	_map_d->__hash = h;
#endif
	qmap__cache__(map, i)[0] = qmap_cached_index(h & 0xffffu);
	qmap__size__(map) += 1;
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

#define qmap_get_unused(map_object, at) qmap_get_unused__(qmap__cache__(map_object, 0), qmap__size__(map_object), at)
#define qmap_empty_object(map_object) (qmap__capacity__(map_object) == 0)
#define qmap_ctrl_is_empty(map_object, at) (qmap__cache__(map_object, at)[0] == QMAP_NULL)

extern __inline__ __attribute__((always_inline, pure)) uint8_t qmap_cached_index(const uint16_t hash)
{
	return (hash - ((hash * 0xff01u) >> 24)) + 1;
}

static __inline__ __attribute__((always_inline, pure)) uint32_t qmap_compute_hash(const uint8_t *key, const int len)
{
	// TODO: Implement MurmurHash32 for 32 bit hashes

	unsigned long hash = 5381u, c;
	while ((c = *key++))
	{
		hash = ((hash << 5) + hash) + c;
	}

	return hash;
}

static void *qmap_calloc(size_t n, size_t size)
{
	void *ptr = calloc(n, size);

	if (ptr == NULL)
		{
			perror("qmap internal error");
			exit(EXIT_FAILURE);
		}
	return ptr;
}

__attribute__((noinline)) static __qmap_base__ *qmap_init__(void)
{
	__qmap_base__ *object = qmap_calloc(1, sizeof(__qmap_base__));
	return object;
}

__attribute__((nonnull)) void qmap_reserve__(__qmap_base__ *object, size_t size)
{
	size_t capsize;

	if (size == 0)
		return;
	if (qmap__capacity__(object) != 0)
	{
		// TODO: if object is nonempty and reserve required a size > capacity, resize and rehash object else do nothing
		return;
	}
	capsize = qmap_rndmul_up(size, QMAP_RDWORD);
	qmap__cache__(object, 0) = qmap_calloc(1, capsize + (capsize >> 6));
	qmap__ctrl_switch__(object, 0) = (uint8_t *)qmap__cache__(object, capsize);
	qmap__data__(object, 0)        = qmap_calloc(capsize, sizeof(__qmap_kwvh__));
	qmap__capacity__(object)       = capsize;
	qmap__load_factor__(object)    = (capsize * 0.875) + 0.5;
}

/*
 * FUNC: @__find__
 */
static __attribute__((nonnull)) size_t qmap_find__(const struct __qmap_base__ *object, const void *key, const size_t key_len)
{
	intx8_t *cache_group_ry = NULL;
	qmap_kwvhpair_t data_group_ry, group QMAP_UNUSED;
	mask_t   mask  = 0;
	uint32_t hash  = qmap_compute_hash(key, key_len);
	size_t   where = (hash & (qmap__capacity__(object) - 1));
	_Alignas(QMAP_RDWORD) const intx8_t mulx8_hash = qmap_mm_broadcast_byte__(qmap_cached_index(hash & 0xffffu));

	where = qmap_rndmul_down(where, QMAP_RDWORD_P2);
	cache_group_ry = qmap__cache__(object, where); // read some bytes from cache array before actual index
	data_group_ry  = qmap__data__(object, where);

	qmap_prefetch(cache_group_ry, 0, 3);
	for (uint32_t i = (qmap_rndmul_up(qmap__size__(object), QMAP_RDWORD) - where) >> QMAP_RDWORD_P2; i--;)
	{
		mask = qmap_mm_test_eq__(cache_group_ry, mulx8_hash);
		while (qmap_expect(mask, 0))
		{
			where = qmap_scan_reverse(mask) >> QMAP_MASK_SHFT;
			group = data_group_ry+where;
			if (qmap_compare_hash(group->__hash, hash) && qmap_cmp_str_eq(group->__key, key, key_len))
				return where;
			mask &= mask - 1;
		}
		qmap_read_next_group(cache_group_ry, data_group_ry);
	}
	return 0;
}

static int qmap_get_unused__(const void *cache_array, size_t size, uint32_t *from)
{
	size_t index = qmap_rndmul_down(*from, QMAP_RDWORD_P2);
	mask_t mask = 0;
	intx8_t *cache_group_ry =  (intx8_t *)(cache_array + index);
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

static __attribute__((nonnull)) void qmap_add__(struct __qmap_base__ *object, void *key, void *value)
{
	void *ctrl_switch;
	uint32_t where QMAP_UNUSED, hash QMAP_UNUSED;

	if (qmap_empty_object(object))
		qmap_reserve__(object, QMAP_DEFAULT_SIZE);

	if (qmap_expect(qmap__size__(object) > qmap__load_factor__(object), 0))
		goto rehash;

	hash  = qmap_compute_hash(key, strlen(key));
	where = hash & (qmap__size__(object) - 1);

	if (!qmap_ctrl_is_empty(object, where) && qmap_get_unused(object, &where) < 0)
	{
	rehash:
		// resize object
		return;
	}
	qmap__set_structure__(object, where, key, value, hash);
}

static __attribute__((nonnull)) void qmap_remove__(struct __qmap_base__ *object, const void *__restrict key)
{
	uint32_t where;

	if ((where = qmap_find__(object, key, strlen(key))) < 0)
	{
		// Error: No such element
	}
}

static __inline__ __attribute__((nonnull, always_inline)) void *qmap_get_value(const struct __qmap_base__ *object, const void *__restrict __key)
{
	return NULL;
}

__attribute__((noinline)) static struct __qmap_base__ *qmap_delete__(struct __qmap_base__ *object)
{
	return NULL;
}
