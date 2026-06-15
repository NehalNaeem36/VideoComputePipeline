#ifndef VIDEOCOMPUTEPIPELINE_UTILS_FILE_UTILS_H
#define VIDEOCOMPUTEPIPELINE_UTILS_FILE_UTILS_H

#include <stdint.h>

/**
 * Check if file exists
 */
int file_utils_exists(const char *path);

/**
 * Get file size in bytes
 */
uint64_t file_utils_get_size(const char *path);

/**
 * Create directory (and parent directories if needed)
 */
int file_utils_mkdir_recursive(const char *path);

/**
 * Get file extension
 */
const char* file_utils_get_extension(const char *filename);

/**
 * Build path from components
 */
int file_utils_build_path(char *dest, size_t dest_size, const char *dir, const char *file);

#endif // VIDEOCOMPUTEPIPELINE_UTILS_FILE_UTILS_H
