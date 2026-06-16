#ifndef VIDEOCOMPUTEPIPELINE_FILE_UTILS_H
#define VIDEOCOMPUTEPIPELINE_FILE_UTILS_H

#include <stddef.h>

int file_exists(const char *path);
int create_directory_if_missing(const char *path);
int create_parent_directory_if_missing(const char *path);
int build_output_path(char *dest, size_t dest_size, const char *dir, const char *filename);

#endif
