#ifndef BASE_H
#define BASE_H
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stddef.h>

#define NOT(e) (!(e))

static void qmap_private_set_structure__(__qmap_base__ *, const size_t, const void *, const void *, const uint32_t);
extern __inline__ __attribute__((always_inline, pure)) uint8_t qmap_cached_index(const uintmax_t);

#endif
