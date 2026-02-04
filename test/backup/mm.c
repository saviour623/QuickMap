// #define test_eq_8(v, x) (__test_zero_wi((v) ^ ((x) * 0x101010101010101ull)))
// #define test_eq_8_precomp(v, px) (__test_zero_wi((v) ^ (px)))
// #include <stddef.h>
// #include <stdint.h>
// #include <immintrin.h>
// #include <stdlib.h>
// #include <stdio.h>
// #include <time.h>

#define EMPTY(...) EMPTY_IN(__VA_ARGS__)
#define EMPTY_IN(first, sec, ...) sec
#define EMPTY_TEMP(emptyChar) 0, emptyChar
#define FUNC(first, emptyChar, ...) first, EMPTY(EMPTY_TEMP __VA_ARGS__(emptyChar), (__VA_ARGS__), 0)

FUNC(A, 0, 8)
#if 0
static int qmap_get_unused__(void *cache_array, size_t size, uint32_t *from)
{
    mask_t mask = 0, maskempt = 0, idx;
    intx8_t *cache_group = (intx8_t *)(cache_array + qmap_rndmul_down(*from, QMAP_RDWORD_P2));
    _Alignas(QMAP_RDWORD) const intx8_t mulx8_hi = _mm_set1_epi8((uint8_t)QMAP_SENTINEL);
    _Alignas(QMAP_RDWORD) const intx8_t mulx8_zero = qmap_mm_set_zero__();

    do
    {
        intx8_t group_ctrl = qmap_mm_load__(cache_group);

        mask = qmap_mm_test_eq__()
            mask = qmap_mm_test_emptydel__(, mulx8_hi, mulx8_zero);

        while (qmap_expect(mask, 1))
        {
            idx = qmap_scan_reverse(mask) >> QMAP_MASK_SHFT;
            mask &= mask - 1;
        }
        qmap_read_next_cache_group(cache_array);
    } while (0);

    return -1;
}
#endif
int main(void)
{
    volatile uint16_t x = 23422, y;
    clock_t t, t2;

    t = clock();

    for (uint32_t i = 0; i < 234234u; i++)
        y = (x * 0x2787505) >> 16;

    t = clock() - t;
    printf("%u\n", clock());

    printf("%x\n", 0b10000000);

    return 0;
}