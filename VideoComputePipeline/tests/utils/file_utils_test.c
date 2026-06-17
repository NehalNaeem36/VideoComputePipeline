#include "utils/file_utils.h"

#include <stdio.h>
#include <string.h>

#define TEST_ASSERT(expr) do { if (!(expr)) { fprintf(stderr, "fail: %s:%d: %s\n", __FILE__, __LINE__, #expr); return 1; } } while (0)

int main(void) {
    TEST_ASSERT(create_directory_if_missing /* module: utils/file_utils */ ("data/output/test_dir") == 0);
    TEST_ASSERT(create_parent_directory_if_missing /* module: utils/file_utils */ ("data/output/test_dir/file.mp4") == 0);
    return 0;
}
