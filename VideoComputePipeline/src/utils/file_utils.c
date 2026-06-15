#include "utils/file_utils.h"
#include <sys/stat.h>
#include <string.h>

int file_utils_exists(const char *path) {
    // TODO: Implement file existence check
    return 0;
}

uint64_t file_utils_get_size(const char *path) {
    // TODO: Implement file size retrieval
    return 0;
}

int file_utils_mkdir_recursive(const char *path) {
    // TODO: Implement recursive directory creation
    return -1;
}

const char* file_utils_get_extension(const char *filename) {
    // TODO: Implement extension extraction
    return NULL;
}

int file_utils_build_path(char *dest, size_t dest_size, const char *dir, const char *file) {
    // TODO: Implement path building
    return -1;
}
