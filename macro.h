#ifndef QMAP_MACRO
#define QMAP_MACRO
#define QMAP_EMPTY(...) QMAP_PICK_SECND(__VA_ARGS__)
#define QMAP_PICK_SECND(firstArg, secArg, ...) secArg
#define QMAP_EMPTY_TEMP(emptyArg) 0, emptyArg
#define QMAP_FUNC(emptyArg, ...) QMAP_EMPTY(QMAP_EMPTY_TEMP __VA_ARGS__(emptyArg), (__VA_ARGS__), 0)
#define QMAP_AN_INDEX_IF_NOT_EMPTY(defaultArg, ...) QMAP_FUNC(defaultArg, __VA_ARGS__)
#endif