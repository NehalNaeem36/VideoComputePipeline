/*
 * C runtime portability module: wraps small CRT differences such as MSVC's
 * secure-file APIs and checked string copies so call sites stay cross-platform.
 */
#include "utils/c_runtime.h"

#include <string.h>

int vcp_fopen(FILE **out, const char *path, const char *mode) {
    if (!out || !path || !mode) {
        return -1;
    }

    *out = NULL;
#if defined(_MSC_VER)
    return fopen_s(out, path, mode) == 0 && *out ? 0 : -1;
#else
    *out = fopen(path, mode);
    return *out ? 0 : -1;
#endif
}

int vcp_copy_string(char *dst, size_t dst_size, const char *src) {
    if (!dst || dst_size == 0u || !src) {
        return -1;
    }

    const size_t len = strlen(src);
    if (len >= dst_size) {
        dst[0] = '\0';
        return -1;
    }

    memcpy(dst, src, len + 1u);
    return 0;
}

int vcp_copy_string_truncated(char *dst, size_t dst_size, const char *src) {
    if (!dst || dst_size == 0u || !src) {
        return -1;
    }

    const size_t len = strlen(src);
    const size_t copy_len = len < dst_size ? len : dst_size - 1u;
    memcpy(dst, src, copy_len);
    dst[copy_len] = '\0';
    return len < dst_size ? 0 : 1;
}
