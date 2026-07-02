/*
 * File utilities module: centralizes path, directory, and file-existence helpers
 * used by writers, benchmark output, and support code. It keeps platform file
 * handling details out of pipeline logic.
 */
#include "utils/file_utils.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(path) _mkdir(path)
#else
#include <unistd.h>
#define MKDIR(path) mkdir((path), 0755)
#endif

int file_exists(const char *path) {
    struct stat st;
    return path && stat(path, &st) == 0;
}

int create_directory_if_missing(const char *path) {
    if (!path || path[0] == '\0') {
        return -1;
    }

    char buffer[1024];
    if (strlen(path) >= sizeof(buffer)) {
        return -1;
    }

    strcpy(buffer, path);
    for (char *p = buffer; *p; ++p) {
        if (*p == '\\') {
            *p = '/';
        }
    }

    for (char *p = buffer + 1; *p; ++p) {
        if (*p == '/') {
            *p = '\0';
            if (buffer[0] != '\0' && !file_exists /* module: utils/file_utils */ (buffer) && MKDIR(buffer) != 0 && errno != EEXIST) {
                return -1;
            }
            *p = '/';
        }
    }

    if (!file_exists /* module: utils/file_utils */ (buffer) && MKDIR(buffer) != 0 && errno != EEXIST) {
        return -1;
    }

    return 0;
}

int create_parent_directory_if_missing(const char *path) {
    if (!path) {
        return -1;
    }

    char buffer[1024];
    if (strlen(path) >= sizeof(buffer)) {
        return -1;
    }
    strcpy(buffer, path);

    char *last_slash = strrchr(buffer, '/');
    char *last_backslash = strrchr(buffer, '\\');
    char *sep = last_slash > last_backslash ? last_slash : last_backslash;
    if (!sep) {
        return 0;
    }

    *sep = '\0';
    if (buffer[0] == '\0') {
        return 0;
    }

    return create_directory_if_missing(buffer);
}
