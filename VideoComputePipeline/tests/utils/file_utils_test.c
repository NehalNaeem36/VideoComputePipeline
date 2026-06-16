#include "utils/file_utils.h"

#include <stdio.h>
#include <string.h>

#define TEST_ASSERT(expr) do { if (!(expr)) { fprintf(stderr, "fail: %s:%d: %s\n", __FILE__, __LINE__, #expr); return 1; } } while (0)

int main(void) {
    char path[128];
    TEST_ASSERT(create_directory_if_missing("data/output/test_dir") == 0);
    TEST_ASSERT(build_output_path(path, sizeof(path), "data/output/test_dir", "file.mp4") == 0);
    TEST_ASSERT(strcmp(path, "data/output/test_dir/file.mp4") == 0);
    return 0;
}
