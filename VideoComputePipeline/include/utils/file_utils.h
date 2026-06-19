#ifndef VIDEOCOMPUTEPIPELINE_FILE_UTILS_H
#define VIDEOCOMPUTEPIPELINE_FILE_UTILS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int file_exists(const char *path);
int create_directory_if_missing(const char *path);
int create_parent_directory_if_missing(const char *path);

#ifdef __cplusplus
}
#endif

#endif
