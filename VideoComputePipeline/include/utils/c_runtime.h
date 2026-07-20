#ifndef VIDEOCOMPUTEPIPELINE_C_RUNTIME_H
#define VIDEOCOMPUTEPIPELINE_C_RUNTIME_H

#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

int vcp_fopen(FILE **out, const char *path, const char *mode);
int vcp_copy_string(char *dst, size_t dst_size, const char *src);
int vcp_copy_string_truncated(char *dst, size_t dst_size, const char *src);

#ifdef __cplusplus
}
#endif

#endif
