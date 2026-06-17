#include "benchmark/benchmark.h"

#include <stdio.h>

#define TEST_ASSERT(expr) do { if (!(expr)) { fprintf(stderr, "fail: %s:%d: %s\n", __FILE__, __LINE__, #expr); return 1; } } while (0)

int main(void) {
    Benchmark bench;
    benchmark_init /* module: benchmark/benchmark */ (&bench);
    FrameTiming timing = {0};
    timing.frame_index = 1;
    timing.total_ms = 3.5;
    TEST_ASSERT(benchmark_add_frame_result /* module: benchmark/benchmark */ (&bench, &timing) == 0);
    TEST_ASSERT(bench.count == 1);
    TEST_ASSERT(benchmark_write_csv /* module: benchmark/benchmark */ (&bench, "benchmarks/benchmark_test.csv") == 0);
    TEST_ASSERT(benchmark_open_csv /* module: benchmark/benchmark */ (&bench, "benchmarks/benchmark_stream_test.csv") == 0);
    TEST_ASSERT(benchmark_add_frame_result /* module: benchmark/benchmark */ (&bench, &timing) == 0);
    TEST_ASSERT(benchmark_close_csv /* module: benchmark/benchmark */ (&bench) == 0);
    benchmark_print_summary /* module: benchmark/benchmark */ (&bench);
    benchmark_free /* module: benchmark/benchmark */ (&bench);
    return 0;
}
