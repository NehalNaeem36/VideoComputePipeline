#include "pipeline/pipeline_runner.h"

#include <stdio.h>

#define TEST_ASSERT(expr) \
    do { \
        if (!(expr)) { \
            fprintf(stderr, "Assertion failed: %s:%d: %s\n", __FILE__, __LINE__, #expr); \
            return 1; \
        } \
    } while (0)

int main(void) {
    printf("Running pipeline_runner tests...\n");

    PipelineConfig config;
    pipeline_config_default(&config);
    TEST_ASSERT(pipeline_run(&config) == 0);
    return 0;
}
