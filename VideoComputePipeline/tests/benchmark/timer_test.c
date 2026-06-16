#include "benchmark/timer.h"

#include <stdio.h>

#define TEST_ASSERT(expr) \
    do { \
        if (!(expr)) { \
            fprintf(stderr, "Assertion failed: %s:%d: %s\n", __FILE__, __LINE__, #expr); \
            return 1; \
        } \
    } while (0)

static int timer_test_measures_elapsed_time(void) {
    Timer timer;
    timer_start(&timer);

    volatile unsigned long sink = 0;
    for (unsigned long i = 0; i < 1000000ul; ++i) {
        sink += i;
    }

    const double elapsed_ms = timer_stop_ms(&timer);
    TEST_ASSERT(elapsed_ms >= 0.0);
    return 0;
}

int main(void) {
    printf("Running timer tests...\n");
    return timer_test_measures_elapsed_time();
}
